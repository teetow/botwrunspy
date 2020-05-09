from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Rect():
    y_start: float
    y_end: float
    x_start: float
    x_end: float

    def as_tuple(self):
        return (self.y_start, self.y_end, self.x_start, self.x_end)

    def scaled(self, w, h):
        return Rect(
            int(self.y_start * h),
            int(self.y_end * h),
            int(self.x_start * w),
            int(self.x_end * w),
        )

    @classmethod
    def full(cls):
        return Rect(0.0, 1.0, 0.0, 1.0)

    @classmethod
    def from_scaled(cls, rect: Rect, w, h):
        return cls(
            rect.y_start / h,
            rect.y_end / h,
            rect.x_start / w,
            rect.x_end / w,
        )
