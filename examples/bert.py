"""
example_grouped_queries_attention.py
"""

import math
import random

import numpy as np

from cattorch.nn.layers import Dropout, Embedding, LayerNorm, Linear
from cattorch.nn.module import Module
from cattorch.optim.optim import Adam
from cattorch.tensor import Tensor

# --------------------------- Toy dataset ---------------------------
SENTENCES = [
    "i like cat",
    "you love dogs",
    "they like dogs and cats",
    "we love machine learning",
    "cats and dogs are friends",
]

# simple binary labels (e.g., does sentence contain 'cats'?)
LABELS = [1 if "cats" in s or "cat" in s else 0 for s in SENTENCES]


# build tiny vocab
def build_vocab(sentences):
    tokens = set()
    for s in sentences:
        for w in s.strip().lower().split():
            tokens.add(w)
    vocab = {w: i + 2 for i, w in enumerate(sorted(tokens))}  # reserve 0=pad,1=unk
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    return vocab


VOCAB = build_vocab(SENTENCES)
INV_VOCAB = {i: w for w, i in VOCAB.items()}

max_len = max(len(s.split()) for s in SENTENCES)


# encode sentences to fixed-length index arrays
def encode(sentences, vocab, max_len):
    arr = []
    for s in sentences:
        toks = s.strip().lower().split()
        idxs = [vocab.get(t, vocab["<unk>"]) for t in toks]
        # pad on right
        idxs = idxs + [vocab["<pad>"]] * (max_len - len(idxs))
        arr.append(idxs)
    return np.array(arr, dtype=np.int32)


X_np = encode(SENTENCES, VOCAB, max_len)  # shape (5, max_len)
Y_np = np.array(LABELS, dtype=np.int64)

# convert to cattorch tensors (indices used directly by Embedding)
X = Tensor(X_np, requires_grad=False)
Y = Tensor(Y_np, requires_grad=False)


# -------------------------
# Small utilities
# -------------------------
def gelu(x: Tensor) -> Tensor:
    """GELU approximation using tanh."""
    # GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
    return (
        0.5
        * x
        * (1.0 + (((x + 0.044715 * (x**3)) * (math.sqrt(2.0 / math.pi))).tanh()))
    )


def _softmax_last_dim(x: Tensor) -> Tensor:
    ex = x.exp()
    denom = ex.sum(dim=-1, keepdims=True)
    return ex / denom


# -------------------------
# FFN
# -------------------------
class FFN(Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = Linear(in_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, in_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, S, in_dim)
        h = self.fc1(x)
        h = gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return h


# -------------------------
# MultiHeadAttention (group heads across d_model)
# -------------------------
class MultiHeadAttention(Module):
    """
    Multi-head attention with optional grouping of adjacent heads (grouping along d_model).

    - d_model: total hidden dim
    - num_heads: number of small heads
    - heads_per_group: number of small heads per meta-head (meta-head dim = head_dim * heads_per_group)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        heads_per_group: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % heads_per_group == 0, (
            "num_heads must be divisible by heads_per_group"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.heads_per_group = heads_per_group
        self.head_dim = d_model // num_heads
        self.num_groups = num_heads // heads_per_group
        self.group_dim = self.head_dim * self.heads_per_group

        # projections
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        x: (B, S, d_model)
        returns: (B, S, d_model)
        """
        B, S, D = x.shape  # noqa: F841
        Q = self.q_proj(x)  # (B, S, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # split into small heads: (B, S, num_heads, head_dim)
        Qh = Q.reshape(B, S, self.num_heads, self.head_dim)
        Kh = K.reshape(B, S, self.num_heads, self.head_dim)
        Vh = V.reshape(B, S, self.num_heads, self.head_dim)

        # bring head axis forward: (B, num_heads, S, head_dim)
        Qh = Qh.transpose(1, 2)
        Kh = Kh.transpose(1, 2)
        Vh = Vh.transpose(1, 2)

        # if heads_per_group == 1 => standard multi-head attention
        if self.heads_per_group == 1:
            # K_t: (B, num_heads, head_dim, S)
            K_t = Kh.transpose(2, 3)
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = (Qh @ K_t) * scale  # (B, num_heads, S, S)

            # softmax over last dim
            weights = _softmax_last_dim(scores)  # (B, num_heads, S, S)

            attn = weights @ Vh  # (B, num_heads, S, head_dim)
            attn = attn.transpose(1, 2).reshape(B, S, self.d_model)  # (B, S, d_model)
            out = self.out_proj(attn)
            out = self.dropout(out)
            return out

        # else: group adjacent heads -> meta-heads
        # reshape to (B, num_groups, heads_per_group, S, head_dim)
        Qg = Qh.reshape(B, self.num_groups, self.heads_per_group, S, self.head_dim)
        Kg = Kh.reshape(B, self.num_groups, self.heads_per_group, S, self.head_dim)
        Vg = Vh.reshape(B, self.num_groups, self.heads_per_group, S, self.head_dim)

        # transpose to (B, num_groups, S, heads_per_group, head_dim) then collapse last two dims -> group_dim
        Qg = Qg.transpose(2, 3).reshape(B, self.num_groups, S, self.group_dim)
        Kg = Kg.transpose(2, 3).reshape(B, self.num_groups, S, self.group_dim)
        Vg = Vg.transpose(2, 3).reshape(B, self.num_groups, S, self.group_dim)

        # attention per group:
        K_t = Kg.transpose(2, 3)  # (B, num_groups, group_dim, S)
        scale = 1.0 / math.sqrt(self.group_dim)
        scores = (Qg @ K_t) * scale  # (B, num_groups, S, S)
        weights = _softmax_last_dim(scores)  # softmax on last dim

        attn_groups = weights @ Vg  # (B, num_groups, S, group_dim)

        # ungroup -> (B, num_groups, S, heads_per_group, head_dim)
        tmp = attn_groups.reshape(
            B, self.num_groups, S, self.heads_per_group, self.head_dim
        )
        # transpose to (B, num_groups, heads_per_group, S, head_dim)
        tmp = tmp.transpose(2, 3)  # (B, num_groups, heads_per_group, S, head_dim)
        # merge groups and heads_per_group -> (B, num_heads, S, head_dim)
        attn_heads = tmp.reshape(
            B, self.num_groups * self.heads_per_group, S, self.head_dim
        )
        # to (B, S, num_heads, head_dim)
        attn_heads = attn_heads.transpose(1, 2)
        # flatten heads -> (B, S, d_model)
        attn_flat = attn_heads.reshape(B, S, self.d_model)

        out = self.out_proj(attn_flat)
        out = self.dropout(out)
        return out


# -------------------------
# Transformer Block
# -------------------------
class TransformerBlock(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        heads_per_group: int = 1,
        ffn_hidden: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if ffn_hidden is None:
            ffn_hidden = 4 * d_model
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model,
            num_heads=num_heads,
            heads_per_group=heads_per_group,
            dropout=dropout,
        )
        self.norm2 = LayerNorm(d_model)
        self.ffn = FFN(d_model, ffn_hidden, dropout=dropout)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # pre-norm style
        y = self.norm1(x)
        y = self.attn(y, mask=mask)
        y = self.dropout(y)
        x = x + y

        z = self.norm2(x)
        z = self.ffn(z)
        z = self.dropout(z)
        x = x + z
        return x


# -------------------------
# Bert-like encoder
# -------------------------
class BertLikeModel(Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        max_len: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        heads_per_group: int = 1,
        ffn_hidden: int = None,
        dropout: float = 0.1,
        use_cls_token: bool = False,
    ):
        super().__init__()
        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(max_len, d_model)
        self.dropout = Dropout(dropout)
        self.layers = []
        for i in range(num_layers):
            block = TransformerBlock(
                d_model,
                num_heads=num_heads,
                heads_per_group=heads_per_group,
                ffn_hidden=ffn_hidden,
                dropout=dropout,
            )
            # register submodule and keep reference
            self.register_module(f"layer_{i}", block)
            self.layers.append(block)
        self.norm = LayerNorm(d_model)
        self.d_model = d_model
        self.max_len = max_len
        self.use_cls_token = use_cls_token

    def forward(self, token_ids: Tensor, mask: Tensor = None) -> Tensor:
        """
        token_ids: (B, S)
        mask: optional boolean mask (B, S) where True => valid token
        returns: (B, S, d_model)
        """
        B, S = token_ids.shape
        assert S <= self.max_len, f"sequence length {S} > max_len {self.max_len}"

        # build pos ids (B, S)
        pos = np.arange(S, dtype=np.int32)
        pos_mat = np.tile(pos[None, :], (B, 1))
        pos_t = Tensor(pos_mat, requires_grad=False)

        tok_emb = self.token_emb(token_ids)  # (B, S, d_model)
        pos_emb = self.pos_emb(pos_t)  # (B, S, d_model)
        x = tok_emb + pos_emb
        x = self.dropout(x)

        for block in self.layers:
            x = block(x, mask=mask)

        x = self.norm(x)
        return x


# -------------------------
# Classifier head
# -------------------------
class ClassifierHead(Module):
    def __init__(
        self,
        d_model: int,
        num_classes: int,
        hidden: int = None,
        dropout: float = 0.1,
        pool: str = "mean",
    ):
        super().__init__()
        self.pool = pool
        if hidden is not None:
            self.fc_hidden = Linear(d_model, hidden)
            self.fc_out = Linear(hidden, num_classes)
            self.dropout = Dropout(dropout)
        else:
            self.fc_hidden = None
            self.fc_out = Linear(d_model, num_classes)
            self.dropout = Dropout(dropout)

    def forward(self, encoder_outputs: Tensor, mask: Tensor = None) -> Tensor:
        """
        encoder_outputs: (B, S, d_model)
        mask: optional boolean mask (B, S) True => valid
        """
        B, S, D = encoder_outputs.shape  # noqa: F841
        if self.pool == "cls":
            pooled = encoder_outputs[:, 0, :]
        elif self.pool == "mean":
            if mask is None:
                pooled = encoder_outputs.mean(dim=1)
            else:
                # mask may be Tensor or numpy; convert to float Tensor
                try:
                    maskf = mask.astype(encoder_outputs.data.asnumpy().dtype)
                except Exception:
                    # fallback: if mask is Tensor-like with .data
                    try:
                        maskf = Tensor(
                            mask.data.asnumpy().astype(np.float32), requires_grad=False
                        )
                    except Exception:
                        maskf = Tensor(mask.astype(np.float32), requires_grad=False)
                maskf = maskf.reshape(B, S, 1)
                summed = (encoder_outputs * maskf).sum(axis=1)  # (B, D)
                counts = maskf.sum(axis=1)  # (B, 1)
                pooled = summed / (counts + 1e-9)
        else:
            raise ValueError("pool must be 'cls' or 'mean'")

        if self.fc_hidden is not None:
            h = self.dropout(pooled)
            h = self.fc_hidden(h)
            h = gelu(h)
            h = self.dropout(h)
            logits = self.fc_out(h)
        else:
            logits = self.fc_out(self.dropout(pooled))
        return logits


# ---------------------------
# Build full model: BertLike + Classifier
# ---------------------------
class SimpleBertClassifier(Module):
    def __init__(
        self,
        vocab_size,
        d_model=64,
        max_len=32,
        num_layers=2,
        num_heads=4,
        heads_per_group=1,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = BertLikeModel(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            num_layers=num_layers,
            num_heads=num_heads,
            heads_per_group=heads_per_group,
            dropout=dropout,
        )
        self.classifier = ClassifierHead(
            d_model=d_model,
            num_classes=num_classes,
            hidden=None,
            dropout=dropout,
            pool="mean",
        )
        self.register_module("encoder", self.encoder)
        self.register_module("classifier", self.classifier)

    def forward(self, token_ids: Tensor, mask: Tensor = None):
        enc = self.encoder(token_ids, mask=mask)  # (B, S, d_model)
        logits = self.classifier(enc, mask=mask)  # (B, num_classes)
        return logits


# --------------------------- Training loop (toy) ---------------------------
def train_toy():
    random.seed(42)
    np.random.seed(42)

    vocab_size = len(VOCAB)
    model = SimpleBertClassifier(
        vocab_size=vocab_size,
        d_model=32,
        max_len=max_len,
        num_layers=2,
        num_heads=4,
        heads_per_group=2,
        num_classes=2,
        dropout=0.1,
    )

    params = list(model.parameters())
    optim = Adam(params, lr=1e-2)

    # convert labels to one-hot (for MSE loss)
    num_classes = 2
    Y_onehot = np.zeros((Y_np.shape[0], num_classes), dtype=np.float32)
    for i, lab in enumerate(Y_np.tolist()):
        Y_onehot[i, lab] = 1.0
    Y_target = Tensor(Y_onehot, requires_grad=False)

    epochs = 300
    for ep in range(epochs):
        logits = model(X)  # (B, num_classes)
        diff = logits - Y_target
        loss = (diff * diff).mean()

        # backward + step
        try:
            loss.backward()
        except Exception:
            # adapt if your Tensor API uses a different backward method
            raise

        optim.step()
        optim.zero_grad()

        if ep % 20 == 0 or ep == epochs - 1:
            try:
                preds = logits.data.asnumpy().argmax(axis=1)
                loss_val = float(loss.data.asnumpy())
            except Exception:
                preds = np.array(logits).argmax(axis=1)
                loss_val = float(np.array(loss))
            acc = (preds == Y_np).mean()
            print(f"epoch {ep:3d} loss={loss_val:.4f} acc={acc:.2f}")

    print("Training finished.")


if __name__ == "__main__":
    train_toy()
