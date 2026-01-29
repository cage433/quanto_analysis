from functools import total_ordering, cached_property

from modelling.utils.type_utils import checked_dict_type

MULT_CACHE = {}
INV_CACHE = {}

@total_ordering
class UOM:
    def __init__(self, powers: dict[str, int]):
        self.powers = checked_dict_type(powers, str, int)

    def __hash__(self):
        return hash(frozenset(self.powers.items()))

    def __eq__(self, other):
        return isinstance(other, UOM) and self.powers == other.powers

    def __lt__(self, other):
        return str(self) < str(other)

    @cached_property
    def __to_string(self):
        pos_symbols = [k for k, v in self.powers.items() if v > 0 and k != '']
        neg_symbols = [k for k, v in self.powers.items() if v < 0 and k != '']

        def to_text(symbols: list[str], negate: bool) -> str:
            terms = []
            for s in sorted(symbols):
                v = self.powers[s]
                if negate:
                    v = -v
                if v == 1:
                    terms.append(s)
                else:
                    terms.append(f"{s}^{v}")
            return " ".join(terms)

        if len(pos_symbols) > 0:
            numerator = to_text(pos_symbols, False)
            if len(neg_symbols) == 0:
                return numerator
            else:
                denominator = to_text(neg_symbols, True)
                return f"{numerator} / {denominator}"

        else:
            if len(neg_symbols) > 0:
                return to_text(neg_symbols, False)
            else:
                return ''


    def __str__(self):
        return self.__to_string

    def __repr__(self):
        return self.__str__()

    def assert_is_ccy(self):
        if not self.is_ccy:
            raise ValueError(f"{self} is not a ccy")

    @property
    def is_ccy(self):
        return self in CURRENCIES

    @property
    def is_fx_uom(self):
        return self.numerator.is_ccy and self.denominator.is_ccy

    @property
    def numerator(self):
        return UOM({k:v for k, v in self.powers.items() if v > 0})

    @property
    def denominator(self):
        return self.inverse.numerator

    @property
    def inverse(self) -> 'UOM':
        if self not in INV_CACHE:
            inverted = {k:-v for k,v in self.powers.items()}
            INV_CACHE[self] = UOM(inverted)
        return INV_CACHE[self]

    def __mul__(self, other):
        key = (self, other)
        if key not in MULT_CACHE:
            merged_powers = {}
            for k, v in self.powers.items():
                merged_powers[k] = merged_powers.get(k, 0) + v
            for k, v in other.powers.items():
                merged_powers[k] = merged_powers.get(k, 0) + v
            merged_powers = {k:v for k, v in merged_powers.items() if v != 0}
            MULT_CACHE[key] = UOM(merged_powers)
        return MULT_CACHE[key]

    def __truediv__(self, other):
        return self * other.inverse

    @staticmethod
    def uom(symbol: str) -> 'UOM':
        return UOM({symbol:1})

USD = UOM.uom('USD')
GBP = UOM.uom('GBP')
EUR = UOM.uom('EUR')

CURRENCIES = {USD, GBP, EUR}

MWH = UOM.uom('MWH')
MT = UOM.uom('MT')
SHARE = UOM.uom('SHARE')
SCALAR = UOM({})
GBPUSD = USD / GBP
