#!/bin/python


"""

(c) Yanagisawa, 1990

                                ┌───────────────────┐
                                │                   │
                                ↓       a           ↑ c
 (Start) First two colors ─── Position ─┬─-Length ──┴────────────── → (End)
            (1)                 ↑  (4)  ↓ (4 ')                  │
                                │  (5)  │                     b  ↑
                                │       ├-2 colors ── More?  ─┬──┤ d
                                │       ↑ (6)        (7)      ↓  │
                                │       └─────────────────────┘  │
                                │                                │
                                └────────────────────────────────┘
"""

import sys
from dataclasses import dataclass

import numpy as np


@dataclass
class PiHeader:
    width: int
    height: int
    mode: int
    ratio: int
    planes: int
    palette: bytes
    comment: str
    editor: str
    reserved_area: bytes
    header_size: int


default_palette = [
    [0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x70, 0xFF],
    [0x70, 0x00, 0x00, 0xFF],
    [0x70, 0x00, 0x70, 0xFF],
    [0x00, 0x70, 0x00, 0xFF],
    [0x00, 0x70, 0x70, 0xFF],
    [0x70, 0x70, 0x00, 0xFF],
    [0x70, 0x70, 0x70, 0xFF],
    [0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0xF0, 0xFF],
    [0xF0, 0x00, 0x00, 0xFF],
    [0xF0, 0x00, 0xF0, 0xFF],
    [0x00, 0xF0, 0x00, 0xFF],
    [0x00, 0xF0, 0xF0, 0xFF],
    [0xF0, 0xF0, 0x00, 0xFF],
    [0xF0, 0xF0, 0xF0, 0xFF],
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
        decoder = PiDecoder(hdr, data)
        data = decoder()
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

    palette_size = 16 if planes == 4 else 256

    p = data[cur + 15:cur + 15 + 3 * palette_size]
    p = np.array(p, dtype=np.uint8).reshape((palette_size, 3))

    if np.all((p == 0)):
        p = np.array(default_palette, dtype=np.uint8).reshape((palette_size, 3))

    p = np.pad(p, ((0, 0), (0, 1)), mode='constant', constant_values=0xFF)

    print("---------------- PI FILE DUMP ----------------")
    print("Comment : %s" % ("None" if len(comment) == 0 else comment))
    print("Mode : %s" % ("Default palette used" if mode & (1 << 7) == 0 else "Palette data omitted"))
    print("Screen ratio : %.2f" % ratio)
    print("Number of planes : %d" % planes)
    print("Saver model : %s" % saver)
    print("Image dimension : (%d, %d)" % (x, y))
    print("----------------------------------------------")
    print()

    cur += 15 + 3 * palette_size

    return PiHeader(x, y, mode, ratio, planes, p, comment, saver, mra, cur)


def from_u8(x):
    return [(x >> (7 - i)) & 1 for i in range(8)]


class PiDecoder:

    def __init__(self, hdr: PiHeader, data: list):
        self.prev_byte = 0
        self.prev_loc = 0
        self.first_loc = 0

        self.hdr = hdr
        self.cursor = 0

        bin_array = [x for v in data[hdr.header_size:] for x in from_u8(v)]

        if bin_array[-32:] == [0 for _ in range(32)]:
            bin_array = bin_array[:-32]

        self.bin_array = bin_array
        self.d_size = len(self.bin_array)

        self.img = []

        colors_nb = 16 if self.hdr.planes == 4 else 256
        self.prev_byte = 0

        self.delta_table = np.zeros((colors_nb, colors_nb), dtype=np.uint8)

        for a in range(colors_nb):
            for b in range(colors_nb):
                self.delta_table[a, b] = (colors_nb + a - b) % colors_nb

        offset_encoding_4 = {
            bytes([1]): (0, 1),
            bytes([0, 0]): (2, 1),
            bytes([0, 1, 0]): (4, 2),
            bytes([0, 1, 1]): (8, 3),
        }

        offset_encoding_8 = {
            bytes([1]): (0, 1),
            bytes([0, 0]): (2, 1),
            bytes([0, 1, 0]): (4, 2),
            bytes([0, 1, 1, 0]): (8, 3),
            bytes([0, 1, 1, 1, 0]): (16, 4),
            bytes([0, 1, 1, 1, 1, 0]): (32, 5),
            bytes([0, 1, 1, 1, 1, 1, 0]): (64, 6),
            bytes([0, 1, 1, 1, 1, 1, 1, 0]): (128, 7),
        }

        self.current_encoding = offset_encoding_4 if self.hdr.planes == 4 else offset_encoding_8

    def __call__(self):
        size = 120
        print(self.bin_array[:size])
        # self.d_size = size

        self.process_delta_seq()
        self.process_rep_seq()

        while self.cursor < self.d_size:
            self.prev_loc = 0

            self.process_delta_seq()
            self.cursor += 1

            if not self.bin_array[self.cursor - 1]:
                self.process_rep_seq()

        return self.normalize_image()

    def process_delta_seq(self):

        color = self.process_delta()
        if self.cursor >= self.d_size:
            return False

        self.img.append(color)

        color = self.process_delta()
        if self.cursor >= self.d_size:
            return False

        self.img.append(color)

        return True

    def process_rep_seq(self):
        location = []

        while isinstance(location, list) and self.cursor < self.d_size:
            location, length = self.process_repeat()

            self.handle_repeat(location, length)

    def normalize_image(self):
        img = [self.hdr.palette[v] for v in self.img]

        img_data = np.array(img[:self.hdr.width * self.hdr.height], dtype=np.uint8)
        img = np.zeros((self.hdr.height * self.hdr.width, 4), dtype=np.uint8)

        if img_data.shape[0] < img.shape[0]:
            img[:img_data.shape[0]] = img_data[:]
        else:
            img[:] = img_data[:img.shape[0]]

        img = img.reshape((self.hdr.height, self.hdr.width, 4))
        return img.transpose([1, 0, 2])

    def process_delta(self):

        max_len = max([len(v) for v in self.current_encoding.keys()])

        i = 0
        while self.cursor + i < self.d_size and \
                bytes(self.bin_array[self.cursor:self.cursor + i]) not in self.current_encoding:
            i += 1

            if i > max_len:
                raise TypeError('Wrong Pi encoding: delta length ' + str(i))

        if self.cursor + i >= self.d_size:
            self.cursor += i

            return

        bin_version = bytes(self.bin_array[self.cursor:self.cursor + i])
        offset, delta_len = self.current_encoding[bin_version]

        if self.cursor + i + delta_len >= self.d_size:
            self.cursor += i + delta_len

            return

        self.cursor += i
        print('cursor += ', i)
        delta_bin = self.bin_array[self.cursor:self.cursor + delta_len]
        delta = 0
        for i, v in enumerate(delta_bin[::-1]):
            delta |= v << i

        delta += offset

        color = self.delta_table[self.prev_byte, delta]  # shift
        self.delta_table[self.prev_byte, 1:delta + 1] = self.delta_table[self.prev_byte, 0:delta]
        self.delta_table[self.prev_byte, 0] = color

        self.prev_byte = color

        self.cursor += delta_len
        print('cursor += ', delta_len)
        print('delta', delta, color)
        return color

    def process_repeat(self):
        loc = self.bin_array[self.cursor:self.cursor+2]

        if loc == [1, 1]:
            loc = self.bin_array[self.cursor:self.cursor+3]

        self.cursor += len(loc)

        if self.cursor + 3 >= self.d_size or self.prev_loc == loc:
            return 0, 0

        self.prev_loc = loc
        i = 0
        while self.cursor + i < self.d_size and self.bin_array[self.cursor + i] != 0:
            i += 1

        offset = 1 << i
        enc_len = i + 1

        length_bin = self.bin_array[self.cursor + enc_len:self.cursor + enc_len + i]
        length = 0
        for i, v in enumerate(length_bin[::-1]):
            length += v << i

        length += offset

        if self.first_loc == 0:
            length -= 1
            self.first_loc = 1
        if length > 10000:
            print(length_bin)
            print(self.bin_array[self.cursor: self.cursor + enc_len])
            # raise ValueError('length too long : ' + str(length))

        self.cursor += enc_len + i

        return loc, length

    def h_0(self, length: int):
        size = 4

        prec_1 = self.img[-1]
        prec_2 = self.img[-2]

        if len(self.img) < 4:
            size = 2
        if prec_1 == prec_2:
            size = 2

        to_repeat = [self.img[i - size] for i in range(size)]
        idx = 0
        for i in range(length * 2):
            self.img.append(to_repeat[idx])

            idx += 1
            idx %= size

    def do_cpy(self, location, length, exception):
        if len(self.img) < location:
            for i in range(length):
                self.img.append(exception[0])
                self.img.append(exception[1])
        else:
            for i in range(length):
                self.img.append(self.img[-location])
                self.img.append(self.img[-location + 1])

    def h_1(self, length: int):
        first_bytes = self.img[:2]
        self.do_cpy(self.hdr.width, length, first_bytes)

    def h_2(self, length: int):
        first_bytes = self.img[:2]
        self.do_cpy(self.hdr.width * 2, length, first_bytes)

    def h_3(self, length: int):
        first_bytes_reversed = self.img[:2]
        first_bytes_reversed.reverse()

        self.do_cpy(self.hdr.width - 1, length, first_bytes_reversed)

    def h_4(self, length: int):
        first_bytes_reversed = self.img[:2]
        first_bytes_reversed.reverse()

        self.do_cpy(self.hdr.width + 1, length, first_bytes_reversed)

    def handle_repeat(self, location: list, length: int):
        if isinstance(location, int):
            print('STOP')
            return

        print('rep:', location, length)
        handler = {
            bytes([0, 0]): self.h_0,
            bytes([0, 1]): self.h_1,
            bytes([1, 0]): self.h_2,
            bytes([1, 1, 0]): self.h_3,
            bytes([1, 1, 1]): self.h_4,
        }

        # if length > 10000:
        #    raise ValueError('length too long : ' + str(length))
        handler[bytes(location)](length)

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


def write_pix_bmp(data: np.array, file_name: str):
    """
    Takes any raw image array of shape (x, y, 4) with last dimension being (R, G, B, A),
    And saves it to a BMP image file.
    The array data type must be np.uint8, implying a max value in array being 255 (0xFF).

    :param data: The raw input image data
    :param file_name: The file to write the BMP image to.
    """
    def to_uint(x, size):
        return [(x >> i) & 0xFF for i in range(0, size * 8, 8)]

    tmp = data.copy()
    data[..., 0] = tmp[..., 2]
    data[..., 2] = tmp[..., 0]
    data = np.flip(data.transpose([1, 0, 2]), axis=0)

    del tmp

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
