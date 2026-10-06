"""Electoral-district codes are not stable across the 2011 boundary.

2011 codes 08–15 are Kymenlaakso, Etelä-Savo, Pohjois-Savo, Pohjois-Karjala,
Vaasa, Keski-Suomi, Oulu and Lappi. From 2015 the same numbers are
Kaakkois-Suomi, Savo-Karjala, Vaasa, Keski-Suomi, Oulu and Lappi.
Codes 01–07 keep the same abbreviation in every file we hold.
"""



def same_numbered_district(year_a: int, code_a: str, year_b: int, code_b: str) -> bool:
    if str(code_a).zfill(2) != str(code_b).zfill(2):
        return False
    code = str(code_a).zfill(2)
    crosses_2011 = (year_a <= 2011) != (year_b <= 2011)
    return not (crosses_2011 and code >= "08")
