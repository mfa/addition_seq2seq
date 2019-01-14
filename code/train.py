from collections import namedtuple
import datetime
from ignite.engine import Engine, Events, State
from ignite.handlers import EarlyStopping, ModelCheckpoint, Timer
from ignite.metrics import Accuracy, Loss
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data import get_data_loader, PAD_IDX
from model import Encoder, Decoder, AttnDecoder
from utils import maybe_cuda

from tensorboardX import SummaryWriter


def get_model(config):
    encoder = maybe_cuda(Encoder(config), cuda=config.cuda)
    if config.attn_method != "disabled":
        decoder = maybe_cuda(AttnDecoder(config), cuda=config.cuda)
    else:
        decoder = maybe_cuda(Decoder(config), cuda=config.cuda)
    return encoder, decoder


def create_summary_writer(data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    return writer


def run_model_on_batch(encoder, decoder, config, use_teacher_forcing, batch_data):
    (x, x_len), (y, y_len) = batch_data

    encoder_output, encoder_hidden = encoder(
        x=maybe_cuda(Variable(x, requires_grad=False), cuda=config.cuda),
        x_len=x_len,
        hidden=None,
    )

    y_var = maybe_cuda(Variable(y, requires_grad=False), cuda=config.cuda)
    target_length = y_var.size()[1]

    decoder_input = maybe_cuda(
        Variable(torch.LongTensor([[0]] * x.size()[0])), cuda=config.cuda
    )
    decoder_hidden = encoder_hidden

    decoder_output_ls = []
    attn_ls = []
    for di in range(target_length):
        decoder_output, decoder_hidden, other_dict = decoder(
            decoder_input, decoder_hidden, encoder_output
        )
        decoder_output_ls.append(decoder_output)
        if "attn" in other_dict:
            attn_ls.append(other_dict["attn"])

        if use_teacher_forcing:
            decoder_input = y_var[:, di].unsqueeze(1)
        else:
            topv, topi = decoder_output.data.topk(1)
            decoder_input = maybe_cuda(Variable(topi.squeeze(1)), cuda=config.cuda)

    full_decoder_output = torch.cat(decoder_output_ls, dim=1)
    return full_decoder_output, attn_ls


def train(config):
    train_dataloader = get_data_loader(config.train_data_path, config)
    val_dataloader = get_data_loader(config.val_data_path, config)

    encoder, decoder = get_model(config)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.learning_rate)

    dt = datetime.datetime.utcnow()
    writer = create_summary_writer(train_dataloader, log_dir=f"logs/{dt}")

    weight = torch.ones(config.vocab_size)
    weight[PAD_IDX] = 0
    criterion = maybe_cuda(nn.NLLLoss(weight), cuda=config.cuda)

    use_teacher_forcing = np.random.uniform(0, 1) < config.use_teacher_forcing_perc

    def train_step(engine, batch):
        encoder.train()
        decoder.train()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        (x, x_len), (y, y_len) = batch
        full_decoder_output, attn_ls = run_model_on_batch(
            encoder=encoder,
            decoder=decoder,
            config=config,
            use_teacher_forcing=use_teacher_forcing,
            batch_data=batch,
        )

        y_var = maybe_cuda(Variable(y, requires_grad=False), cuda=config.cuda)
        batch_loss = criterion(
            full_decoder_output.view(-1, full_decoder_output.size()[2]), y_var.view(-1)
        )
        batch_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        return batch_loss

    def validate_step(engine, batch):
        encoder.eval()
        decoder.eval()

        (x, x_len), (y, y_len) = batch
        full_decoder_output, attn_ls = run_model_on_batch(
            encoder=encoder,
            decoder=decoder,
            config=config,
            use_teacher_forcing=False,
            batch_data=batch,
        )

        y_var = maybe_cuda(Variable(y, requires_grad=False), cuda=config.cuda)

        batch_loss = criterion(
            full_decoder_output.view(-1, full_decoder_output.size()[2]), y_var.view(-1)
        )
        return batch_loss

    def inspect_step(engine, batch):
        (x, x_len), (y, y_len) = batch
        num = 0

        # We only inspect one element (default: first) of the batch
        single_elem_batch_data = (
            (x[num : num + 1], x_len[num : num + 1]),
            (y[num : num + 1], y_len[num : num + 1]),
        )

        encoder.eval()
        decoder.eval()

        full_decoder_output, attn_ls = run_model_on_batch(
            encoder=encoder,
            decoder=decoder,
            config=config,
            use_teacher_forcing=False,
            batch_data=single_elem_batch_data,
        )

        topv, topi = full_decoder_output.data.topk(1)
        pred_y = topi.squeeze(2)

        mapper = lambda x: val_dataloader.dataset.input_sequence.batch_tensor_to_string(
            x
        )
        print("Input string:\n    {}\n".format(mapper(x)[0]))
        print("Expected output:\n    {}\n".format(mapper(y)[0]))
        print("Predicted output:\n    {}\n".format(mapper(pred_y)[0]))
        return

    trainer = Engine(train_step)
    train_len = len(train_dataloader)

    evaluator = Engine(validate_step)
    inspect = Engine(inspect_step)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        print(
            json.dumps(
                {
                    "metric": "train_loss",
                    "value": float(engine.state.output),
                    "step": engine.state.epoch,
                }
            )
        )
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_dataloader)
        print(
            json.dumps(
                {
                    "metric": "val_loss",
                    "value": float(evaluator.state.output),
                    "step": engine.state.epoch,
                }
            )
        )
        writer.add_scalar(
            "validation/loss", evaluator.state.output, engine.state.iteration
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_inspect_results(engine):
        inspect.run(val_dataloader)

    # early stopping
    def score_function(engine):
        val_loss = engine.state.output
        return -val_loss

    # stop if 10 epochs are worse than before
    handler = EarlyStopping(patience=20, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    # save model if better than last one
    checkpoint_handler = ModelCheckpoint(
        "models",
        "torch_add",
        score_function=score_function,
        n_saved=1,
        require_empty=False,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        checkpoint_handler,
        to_save={"encoder": encoder, "decoder": decoder},
    )

    # timer
    timer = Timer(average=True)
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        print(
            f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]"
        )
        timer.reset()

    trainer.run(train_dataloader, max_epochs=config.epochs)

    writer.close()
