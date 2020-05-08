from __future__ import annotations

import re
from dataclasses import dataclass

import parse

re_name = re.compile(r"(?:(?P<koroks>\d+)[,.][ ]?)?(?P<shrine>.+)")
re_line = re.compile(r"(?P<name>.+?)[ ]?(?P<diff>[-+]\d+(?:[,.:]\d+)?)? (?P<ts>(?:\d+[.,:])?\d+[,.:]\d+)")


def prettify_name(namestr):
    namestr = str.strip(namestr)
    m = re_name.match(namestr)
    if not m:
        return namestr
    d = m.groupdict()
    koroks = d["koroks"]
    shrine = d["shrine"]
    return koroks, shrine


class Split():
    koroks: str
    name: str
    diff: str
    ts: str

    def __init__(self, koroks, name, diff, ts):
        super().__init__()
        self.name = name
        self.koroks = koroks
        self.diff = diff
        self.ts = ts

    @classmethod
    def fromRaw(cls, rawstr) -> Split:
        matches = re_line.match(rawstr)
        if not matches:
            return
        parts = matches.groupdict()
        diff = parts["diff"]
        ts = parts["ts"]
        name = parts["name"]
        koroks, name = prettify_name(name)
        if name and ts:
            return Split(koroks, name, diff, ts)

    @property
    def full_name(self):
        return f"{self.koroks}. {self.name}"

    def set_full_name(self, name: str):
        self.korok, self.name = parse.parse("{}. {}", name)

    def __str__(self):
        out = ""
        if self.koroks:
            out += f"{self.koroks} "
        out += f"{self.name} "
        if self.diff:
            out += f"{self.diff} "
        out += self.ts
        return out


def test():
    s = """
    Plateau Tower - .
    3. Bombs +47.8 20:51
    Paraglider +64 22:48
    10.Wahgo Katta -29.8 32:13
    Dueling Peaks To...-35.4 40:39
    23.Shee Vaneer -1:10 45:13
    24,Shee Venath -1:13 46:34
    25.ReeDahee -1:04 49:01

    25.HaDahamar -1:05 50:42

    30. Daka Tuss 1:05:58
    35. Qukah Nata 1:27:11
    44, Kam Urog 1:40:42"""

    for line in s.split("\n"):
        sp = Split.fromRaw(line)
        if sp:
            print(sp)


if __name__ == "__main__":
    test()
