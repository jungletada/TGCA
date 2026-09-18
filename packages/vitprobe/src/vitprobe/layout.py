"""Explicit token layout; extra-token order is cls -> class -> register.

Factories are convenience metadata only: no model imports or behavior changes.
Sliced diagnostics accept tensors directly; this object centralizes all slicing.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenLayout:
    """A contiguous patch grid and an ordered block of extra tokens.

    class_first=True places ALL extra tokens before patches; False places them
    after patches without reversing their order. cls and semantic class queries
    are distinct slices; neither implicitly includes register tokens.
    """
    n_class: int = 0
    n_cls: int = 0
    n_register: int = 0
    grid_hw: tuple = (0, 0)
    class_first: bool = True

    def __post_init__(self):
        counts = (self.n_cls, self.n_class, self.n_register)
        if any(type(n) is not int or n < 0 for n in counts):
            raise ValueError('Token counts must be nonnegative integers')
        if (not isinstance(self.grid_hw, tuple) or len(self.grid_hw) != 2
                or any(type(n) is not int or n < 0 for n in self.grid_hw)):
            raise ValueError('grid_hw must be a pair of nonnegative integers')
        if (self.grid_hw[0] == 0) != (self.grid_hw[1] == 0):
            raise ValueError('Empty grid must be (0, 0)')
        if type(self.class_first) is not bool:
            raise ValueError('class_first must be boolean')

    @property
    def n_patch(self):
        return self.grid_hw[0] * self.grid_hw[1]

    @property
    def n_extra(self):
        return self.n_cls + self.n_class + self.n_register

    @property
    def n_tokens(self):
        return self.n_extra + self.n_patch

    @property
    def patch_slice(self):
        start = self.n_extra if self.class_first else 0
        return slice(start, start + self.n_patch)

    @property
    def cls_slice(self):
        start = 0 if self.class_first else self.n_patch
        return slice(start, start + self.n_cls)

    @property
    def class_slice(self):
        start = self.cls_slice.stop
        return slice(start, start + self.n_class)

    @property
    def register_slice(self):
        start = self.class_slice.stop
        return slice(start, start + self.n_register)

    @classmethod
    def mctformer_plus(cls, n_class=20, grid=(28, 28)):
        """Named compatibility factory retained by explicit user approval."""
        return cls(n_class=n_class, grid_hw=grid)

    @classmethod
    def dinov3(cls, grid=(28, 28)):
        """Layout example with one cls and four registers, not a model loader."""
        return cls(n_cls=1, n_register=4, grid_hw=grid)

    def patch_tokens(self, tokens):
        """Slice [...,T,D] without changing dtype or gradients."""
        if tokens.ndim < 2 or tokens.shape[-2] != self.n_tokens:
            raise ValueError('Token sequence does not match layout')
        return tokens[..., self.patch_slice, :]

    def patch_grid(self, tokens):
        """Reshape [...,T,D] to [...,H,W,D] in original spatial order."""
        patches = self.patch_tokens(tokens)
        return patches.reshape(*patches.shape[:-2], *self.grid_hw, patches.shape[-1])

    def patch_attention(self, attention):
        """Slice [...,T,T] to patch-query/patch-key submatrix; no renormalization."""
        self.validate_attention(attention)
        return attention[..., self.patch_slice, self.patch_slice]

    def query_patch_attention(self, attention, query_group='class'):
        """Slice a named extra query group, preserving heads and raw group mass."""
        self.validate_attention(attention)
        groups = {'class': self.class_slice, 'cls': self.cls_slice,
                  'register': self.register_slice}
        if query_group not in groups:
            raise ValueError('query_group must be class, cls or register')
        return attention[..., groups[query_group], self.patch_slice]

    def validate_attention(self, attention):
        if attention.ndim < 2 or attention.shape[-2:] != (self.n_tokens, self.n_tokens):
            raise ValueError('Self-attention matrix does not match layout')
