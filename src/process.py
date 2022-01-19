import pickle
import csv
import numpy as np
from string import digits
import re
pitchhz, keynum = {}, {}

keys_s = ('a', 'a#', 'b',  'c',  'c#', 'd', 'd#', 'e',  'f',  'f#', 'g', 'g#')
keys_f = ('a', 'bb', 'b',  'c',  'db', 'd', 'eb', 'e',  'f',  'gb', 'g', 'ab')
keys_e = ('a', 'bb', 'cb', 'b#', 'db', 'd', 'eb', 'fb', 'e#', 'gb', 'g', 'ab')


def getfreq(pr=False):
    if pr:
        print("Piano key frequencies (for equal temperament):")
        print("Key number\tScientific name\tFrequency (Hz)")
    for k in range(88):
        freq = 27.5 * 2.**(k/12.)
        oct = (k + 9) // 12
        note = '%s%u' % (keys_s[k % 12], oct)
        if pr:
            print("%10u\t%15s\t%14.2f" % (k+1, note.upper(), freq))
        pitchhz[note] = freq
        keynum[note] = k
        note = '%s%u' % (keys_f[k % 12], oct)
        pitchhz[note] = freq
        keynum[note] = k
        note = '%s%u' % (keys_e[k % 12], oct)
        pitchhz[note] = freq
        keynum[note] = k
    return pitchhz, keynum


pitch, keyno = getfreq()
filename1 = "./Guitar_13.csv"


def getData(filename1):
    with open(filename1, encoding='utf-8') as csv1:
        reader1 = csv.reader(csv1)
        for row1 in reader1:
            if row1[1].lower() not in pitch:
                row1[1] = re.sub(r'[^A-Za-z0-9]+', '', row1[1])

                # if row1[1].lower() not in pitch:
                #     row1[1] = re.sub(r'#', '', row1[1])
            yield (np.array([row1[1].lower(), 4], dtype=np.str))


            # This will give arrays of floats, for other types change dtype
g_no = []
for g in getData(filename1):
    g_no.append(g)
with open('./Guitar_13.pkl', 'wb') as output:
    pickle.dump(g_no, output)
