#!/bin/python


"""
┌───────────┐
 ｜ ｜
 ↓ a ↑ c
 (Start) First two colors ─── Position ─┬─-Length ──┴────────── → (End)
 (1) ↑ (4) ｜ (4 ') ↑
 ｜ (5) ↓ b ｜
 ｜ ├-2 colors ── More?  ─┬──┤d
 ｜ ↑ (6) (7) ↓ ↓
 ｜ └────────┘ ｜
 ｜ ｜
└────────────────┘
"""

import sys
from dataclasses import dataclass

import numpy as np


@dataclass
class PiHeader:
    x: int
    y: int
    mode: int
    ratio: int
    palette: bytes
    comment: str
    editor: str
    reserved_area: bytes
    header_size: int


default_palette = [
    [0x00, 0x00, 0x00],
    [0x00, 0x00, 0x70],
    [0x70, 0x00, 0x00],
    [0x70, 0x00, 0x70],
    [0x00, 0x70, 0x00],
    [0x00, 0x70, 0x70],
    [0x70, 0x70, 0x00],
    [0x70, 0x70, 0x70],
    [0x00, 0x00, 0x00],
    [0x00, 0x00, 0xF0],
    [0xF0, 0x00, 0x00],
    [0xF0, 0x00, 0xF0],
    [0x00, 0xF0, 0x00],
    [0x00, 0xF0, 0xF0],
    [0xF0, 0xF0, 0x00],
    [0xF0, 0xF0, 0xF0],
]


def main():
    for pifile in sys.argv[1:]:
        with open(pifile, 'rb') as inp:
            data = list(inp.read())

        if bytes(data[0:2]).decode() != "Pi":
            print("%s: Not a PI file!" % pifile)
            continue

        print(f'Converting file {pifile}...')
        print()

        hdr = get_header(data)
        data = as_pix_array(data, hdr)
        result_file = pifile[:pifile.rindex('.')] + '.bmp'
        write_pix_bmp(data, result_file)


def get_header(data):
    cur = 2

    comment = ""
    while data[cur] != 0x1A:
        comment += chr(data[cur])
        cur += 1

    while data[cur] != 0:
        cur += 1

    mode, n, m, planes = data[cur + 1:cur + 5]

    ratio = 1
    if n != 0 or m != 0:
        ratio = n / m

    saver = "".join([chr(c) for c in data[cur + 5:cur + 9]])
    mra = data[cur + 10:cur + 11]
    x = (data[cur + 11] << 8) | data[cur + 12]
    y = (data[cur + 13] << 8) | data[cur + 14]

    p = data[cur + 15:cur+15+1*3*16]

    print("---------------- PI FILE DUMP ----------------")
    print("Comment : %s" % ("None" if len(comment) == 0 else comment))
    print("Mode : %s" % ("Default palette used" if mode & (1 << 7) == 0 else "Palette data omitted"))
    print("Screen ratio : %.2f" % ratio)
    print("Number of planes : %d" % planes)
    print("Saver model : %s" % saver)
    print("Image dimension : (%d, %d)" % (x, y))
    print("----------------------------------------------")
    print()

    cur += 15+1*3*16

    return PiHeader(x, y, mode, ratio, p, comment, saver, mra, cur)


def as_pix_array(data: list, hdr: PiHeader):
    img = np.zeros((hdr.y, hdr.x, 4), dtype=np.uint8)

    """ Pixels are handled in horizontal 2 dot units.

     1) First, record the first (upper left) 2-dot color.
     Then, it is assumed that the two dots fill two lines above the target screen.
     2) Then record horizontally from top left to bottom right (the right end is connected to the left end of the next line).
     3) First of all, there are 2 dots that we are paying attention to, on the other hand, upper right, upper left, upper, 2 upper, left,
        Check how many (2 dot units) the same pattern continues from the 5 places to the left of 2.
        I will.  ~~
     4) Then record the position that makes the longest, and then record the length.
     Then move the point of interest, and if it is not the end, go back to (3) and repeat.
    5) If the same pattern does not continue from any position in (3), record the same position as the previous time.
     I will.  (It cannot be the same position continuously)
     6) Record the color of the point of interest (2 dots) and move the point of interest to the next.  (Move by 2 dots)
     7) Examine the pattern as in (3), and if you cannot find the same pattern, 1 (1
        (Bit length), repeat (6), and if found, record 0 (1 bit length)
        Move to (4).
       (Without 1-bit continuous data, return to (3) or record the length where the pattern cannot be found.
     There may be a way to do it)
     """

    """ a) Down if the position is the same as the previously recorded position.
     b) When "Continue?" is 0, it is right. When it is 1, it is down.
     c) Right when compression is done, up if not.
     d) Up when compression is complete, down if not.
    The end is when the pixel extends out of the target range.
    
     ○ Position
     ・ The following 5 locations are used relative to the point of interest.
     Position 0) A little special.  :-)
    
     Compare the two dots one unit before.
    
     2 dots left for same color 
     4 dots left for different colors
    
             will do.  This is a 4x4 tile measure.
     In most cases, the compression rate is higher when only 2 dots are left,
     Although 4x4 is small, it cannot be ignored.  Also located at 6 places
     If is increased, the coding length for that will not be increased and it will not work.
     You can also scan the image first and divide it into modes
     Actually, there is not much difference, so we do not divide into modes.
     (I personally want to avoid mode classification)
     However, I denied that it was awkward because I did something special.
             Can not.  (; _;)
    
     Position 1) 2 dots on 1 line
     Position 2) 2 dots on 2 lines
     Position 3) 1 dot on 1 line 2 dots to the right
     Position 4) 1 dot on 1 line 2 dots to the left
    
      note)
     In "Overall Flow" (5), I wrote that it cannot be the same position as last time.
     Because the special thing was done in position 0, the same position appears in position 0
         There is a possibility.  So if you set a flag and the last time was position 0, this time
     Make sure that position 0 cannot be reached.  (^^)
    
     ・ Position coding method
     (Maybe there is a way to mix it to length)
    
     Position 0 → 00
     Position 1 → 01
     Position 2 → 10
     Position 3 → 110
     Position 4 → 111
     (0/1 binary code length)
     ○ Length
     ・ The following methods are used.  There may be a bit more statistical encoding
     Not so bad.  :-)
    10
     2-3 3x
     4-7 110xx
     8-15 1110xxx
     16-31 11110xxxx
    
     (X is 0 or 1 in binary)
     And so on
    
     ○ Continuation
     ・ It becomes a flag to continue recording the color.
     Continued → 1
     No continuation → 0
     (Record is 1 bit long)
    
     ○ Color
     -The color is 2 dots per unit, but recording is done separately for each dot.
     The method encodes the number of colors that appeared before.  For example, if it is the same color as last time, it will be 0,
     If it is the same as the previous time, it will be 1.  Then, further add this to the color of the dot to the left.
     Divided into 16 ways.  This is because the closer the color is, the more likely it is that it will reappear.
     Due to the relationship, we use that there is a strong correlation with the color to the left of the dot.
    
     ・ There are various methods.  This time, use the following method.
    
     First, prepare the following table.  (initial value)
    
     (1 dot left color) (old) (new)
     0 123456789ABCDEF0
     1 23456789ABCDEF01
     2 3456789ABCDEF012
     3 456789ABCDEF0123
     4 56789ABCDEF01234
     5 6789ABCDEF012345
     6 789ABCDEF0123456
     7 89ABCDEF01234567
     89 ABCDEF012345678
     9 ABCDEF0123456789
     A BCDEF 0123456789A
     B CDEF 0123456789AB
     C DEF 0123456789ABC
     D EF0123456789 ABCD
     EF0123456789ABCDE
     F 0123456789ABCDEF
    
     (0 to F indicate a color code in 1-digit hexadecimal)
    
     Here, suppose you want to record color 8 (the color to the left of one dot is 5).  Because 1 dot left is 5,
    
     5 6789ABCDEF012345
    
     Look at it.  Then I will record 13 because it is the 13th from the newest one.  And
     Since we bring color 8 to the latest, the table is as follows.
    
     5 679ABCDEF0123458
    
     In other words, bring 8 to the latest position, and after that shift it in sequence.
     After that, repeat the same thing.
     """
    return img


def write_pix_bmp(data: np.array, file_name: str):
    def to_uint(x, size):
        return [(x >> i) & 0xFF for i in range(0, size * 8, 8)]

    hdr = [0x42, 0x4D]  # magic
    info = []
    pixels = data.ravel().tolist()

    offset = 14 + 40
    data_len = data.size + offset
    hdr += to_uint(data_len, 4)  # size
    hdr += [0x00, 0x00]  # _app1
    hdr += [0x00, 0x00]  # _app2
    hdr += to_uint(offset, 4)  # offset

    info += to_uint(40, 4)  # info_hdr_size
    info += to_uint(data.shape[1], 4)  # width
    info += to_uint(data.shape[0], 4)  # height
    info += to_uint(1, 2)  # planes
    info += to_uint(32, 2)  # bits_per_pixel
    info += to_uint(0, 4)  # compression
    info += to_uint(data.size, 4)  # raw_data_size
    info += to_uint(0, 4)  # h_resolution
    info += to_uint(0, 4)  # v_resolution
    info += to_uint(0, 4)  # palette_size
    info += to_uint(0, 4)  # important_colors

    print(f'Writing to output file: {file_name}')

    full_file = bytes(hdr + info + pixels)
    with open(file_name, 'wb') as f:
        f.write(full_file)


if __name__ == "__main__":
    main()