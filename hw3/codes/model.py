import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from rnn_cell import RNNCell, GRUCell, LSTMCell


class RNN(nn.Module):
    def __init__(
            self,
            num_embed_units,  # pretrained wordvec size
            num_units,  # RNN units size
            num_layers,  # number of RNN layers
            num_vocabs,  # vocabulary size
            wordvec,  # pretrained wordvec matrix
            dataloader,  # dataloader
            cell_type="GRU",  # cell type,
            layer_norm=False,
            residual=False,
    ):

        super().__init__()

        # load pretrained wordvec
        self.num_vocabs = num_vocabs
        self.num_units = num_units
        self.wordvec = nn.Embedding.from_pretrained(wordvec)
        # the dataloader
        self.dataloader = dataloader

        # TODO START
        # fill the parameter for multi-layer RNN
        assert (num_layers >= 1)
        if cell_type == "RNN":
            self.cells = nn.Sequential(RNNCell(num_embed_units, num_units),
                                       *[RNNCell(num_units, num_units) for _ in range(num_layers - 1)])
        elif cell_type == "GRU":
            self.cells = nn.Sequential(GRUCell(num_embed_units, num_units),
                                       *[GRUCell(num_units, num_units) for _ in range(num_layers - 1)])
        elif cell_type == "LSTM":
            self.cells = nn.Sequential(LSTMCell(num_embed_units, num_units),
                                       *[LSTMCell(num_units, num_units) for _ in range(num_layers - 1)])
        else:
            raise NotImplementedError("Unknown Cell Type")
        self.cell_type = cell_type
        # TODO END

        # intialize other layers
        self.linear = nn.Linear(num_units, num_vocabs)
        self.layer_norms = nn.Sequential(nn.LayerNorm(num_embed_units), *[nn.LayerNorm(
            num_units) for _ in range(num_layers - 1)]) if layer_norm else None
        self.residual = residual

    def maskNLLLoss(self, logits: torch.tensor, gts: torch.tensor, device):
        # Reference: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
        # Reference: https://blog.csdn.net/uhauha2929/article/details/83019995
        mask = (gts != 0).view(-1, 1)
        logits = F.softmax(logits, dim=1)
        crossEntropy = -torch.log(torch.gather(logits, 1, gts.unsqueeze(1)))
        loss = crossEntropy.masked_select(mask).sum()
        loss = loss.to(device)
        return loss

    def get_gts(self, label: torch.tensor, device):
        batch_size = label.size(0)
        return torch.zeros((batch_size, self.num_vocabs), device=device).scatter_(1, label.unsqueeze(1), 1)

    def forward(self, batched_data, device):
        # Padded Sentences
        sent = torch.tensor(batched_data["sent"],
                            dtype=torch.long,
                            device=device)  # shape: (batch_size, length)
        # An example:
        #   [
        #   [2, 4, 5, 6, 3, 0],   # first sentence: <go> how are you <eos> <pad>
        #   [2, 7, 3, 0, 0, 0],   # second sentence:  <go> hello <eos> <pad> <pad> <pad>
        #   [2, 7, 8, 1, 1, 3]    # third sentence: <go> hello i <unk> <unk> <eos>
        #   ]
        # You can use self.dataloader.convert_ids_to_sentence(sent[0]) to translate the first sentence to string in this batch.

        # Sentence Lengths
        length = torch.tensor(batched_data["sent_length"],
                              dtype=torch.long,
                              device=device)  # shape: (batch)
        # An example (corresponding to the above 3 sentences):
        #   [5, 3, 6]

        batch_size, seqlen = sent.shape

        # TODO START
        # implement embedding layer
        # shape: (batch_size, length, num_embed_units)
        embedding = self.wordvec(sent)
        # TODO END

        now_state = []
        for cell in self.cells:
            now_state.append(cell.init(batch_size, device))

        loss = 0
        logits_per_step = []
        for i in range(seqlen - 1):
            hidden = embedding[:, i]
            for j, cell in enumerate(self.cells):
                # shape: (batch_size, num_units)
                if self.layer_norms is not None:
                    hidden = self.layer_norms[j](hidden)
                new_hidden, now_state[j] = cell(hidden, now_state[j])
                if self.residual and hidden.size(1) == new_hidden.size(1):
                    hidden = new_hidden + hidden
                else:
                    hidden = new_hidden
            logits = self.linear(hidden)  # shape: (batch_size, num_vocabs)
            logits_per_step.append(logits)
        # TODO START
        # calculate loss
        for step, logits in enumerate(logits_per_step):
            # Teacher Forcing
            loss += self.maskNLLLoss(logits, sent[:, step + 1], device)
        loss /= length.sum().item()
        # TODO END
        return loss, torch.stack(logits_per_step, dim=1)

    def top_p_filtering(self, logits, max_probability=1.0, filter_value=-1e20):
        # Reference: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=1)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=1), dim=1)
        sorted_indices_to_remove = cumulative_probs > max_probability
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        batch_size = logits.size(0)
        for i in range(batch_size):
            indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
            logits[i, indices_to_remove] = filter_value
        return logits

    def inference(self, batch_size, device, decode_strategy, temperature,
                  max_probability):
        # First Tokens is <go>
        now_token = torch.tensor([self.dataloader.go_id] * batch_size,
                                 dtype=torch.long,
                                 device=device)
        flag = torch.tensor([1] * batch_size, dtype=torch.float, device=device)

        now_state = []
        for cell in self.cells:
            now_state.append(cell.init(batch_size, device))

        generated_tokens = []
        for _ in range(50):  # max sentecne length
            # TODO START
            # translate now_token to embedding
            # shape: (batch_size, num_embed_units)
            embedding = self.wordvec(now_token)
            # TODO END

            hidden = embedding
            for j, cell in enumerate(self.cells):
                if self.layer_norms is not None:
                    hidden = self.layer_norms[j](hidden)
                new_hidden, now_state[j] = cell(hidden, now_state[j])
                if self.residual and hidden.size(1) == new_hidden.size(1):
                    hidden = new_hidden + hidden
                else:
                    hidden = new_hidden
            logits = self.linear(hidden)  # shape: (batch_size, num_vocabs)

            if decode_strategy == "random":
                prob = (logits / temperature).softmax(
                    dim=-1)  # shape: (batch_size, num_vocabs)
                now_token = torch.multinomial(prob,
                                              1)[:, 0]  # shape: (batch_size)
            elif decode_strategy == "top-p":
                # TODO START
                # implement top-p samplings
                prob = (self.top_p_filtering(logits, max_probability=max_probability)).softmax(
                    dim=-1)  # shape: (batch_size, num_vocabs)
                now_token = torch.multinomial(
                    prob, 1)[:, 0]  # shape: (batch_size)
                # TODO END
            elif decode_strategy == "top-1":
                # implement top-1 samplings
                prob = (logits / temperature).softmax(
                    dim=-1)  # shape: (batch_size, num_vocabs)
                now_token = torch.argmax(prob, 1)  # shape: (batch_size)
            else:
                raise NotImplementedError("unknown decode strategy")

            generated_tokens.append(now_token)
            flag = flag * (now_token != self.dataloader.eos_id)

            if flag.sum().tolist(
            ) == 0:  # all sequences has generated the <eos> token
                break
        return torch.stack(generated_tokens, dim=1).detach().cpu().numpy()
