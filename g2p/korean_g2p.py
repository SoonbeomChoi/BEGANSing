import sys
sys.path.append('../')

import re
import math

ONS = ['k0', 'kk', 'nn', 't0', 'tt', 'rr', 'mm', 'p0', 'pp',
       's0', 'ss', 'oh', 'c0', 'cc', 'ch', 'kh', 'th', 'ph', 'h0']
NUC = ['aa', 'qq', 'ya', 'yq', 'vv', 'ee', 'yv', 'ye', 'oo', 'wa',
       'wq', 'wo', 'yo', 'uu', 'wv', 'we', 'wi', 'yu', 'xx', 'xi', 'ii']
COD = ['', 'kf', 'kk', 'ks', 'nf', 'nc', 'nh', 'tf',
       'll', 'lk', 'lm', 'lb', 'ls', 'lt', 'lp', 'lh',
       'mf', 'pf', 'ps', 's0', 'ss', 'oh', 'c0', 'ch',
       'kh', 'th', 'ph', 'h0', 'ng']
RCD = ['kf', 'nf', 'tf', 'll', 'mf', 'pf', 'ng']

def readfileUTF8(fname):
    f = open(fname, 'r')
    corpus = []

    while True:
        line = f.readline()
        line = line.encode("utf-8")
        line = re.sub(u'\n', u'', line)
        if line != u'':
            corpus.append(line)
        if not line: break

    f.close()
    return corpus

def readRules(rule_book):
    f = open(rule_book, 'r',encoding="utf-8")

    rule_in = []
    rule_out = []

    while True:
        line = f.readline()
        line = re.sub('\n', '', line)

        if line != u'':
            if line[0] != u'#':
                IOlist = line.split('\t')
                rule_in.append(IOlist[0])
                if IOlist[1]:
                    rule_out.append(IOlist[1])
                else:   # If output is empty (i.e. deletion rule)
                    rule_out.append(u'')
        if not line: break
    f.close()

    return rule_in, rule_out

def isHangul(charint):
    hangul_init = 44032
    hangul_fin = 55203
    return charint >= hangul_init and charint <= hangul_fin

def checkCharType(var_list):
    #  1: whitespace
    #  0: hangul
    # -1: non-hangul
    checked = []
    for i in range(len(var_list)):
        if var_list[i] == 32:   # whitespace
            checked.append(1)
        elif isHangul(var_list[i]): # Hangul character
            checked.append(0)
        else:   # Non-hangul character
            checked.append(-1)
    return checked

def graph2phone(graphs):
    # Encode graphemes as utf8
    try:
        graphs = graphs.decode('utf8')
    except AttributeError:
        pass

    integers = []
    for i in range(len(graphs)):
        integers.append(ord(graphs[i]))

    # Romanization (according to Korean Spontaneous Speech corpus; 성인자유발화코퍼스)
    phones = ''

    # Pronunciation
    idx = checkCharType(integers)
    iElement = 0
    while iElement < len(integers):
        if idx[iElement] == 0:  # not space characters
            base = 44032
            df = int(integers[iElement]) - base
            iONS = int(math.floor(df / 588)) + 1
            iNUC = int(math.floor((df % 588) / 28)) + 1
            iCOD = int((df % 588) % 28) + 1

            s1 = '-' + ONS[iONS - 1]  # onset
            s2 = NUC[iNUC - 1]  # nucleus

            if COD[iCOD - 1]:  # coda
                s3 = COD[iCOD - 1]
            else:
                s3 = ''
            tmp = s1 + s2 + s3
            phones = phones + tmp

        elif idx[iElement] == 1:  # space character
            tmp = '#'
            phones = phones + tmp

        phones = re.sub('-(oh)', '-', phones)
        iElement += 1
        tmp = ''

    # 초성 이응 삭제
    phones = re.sub('^oh', '', phones)
    phones = re.sub('-(oh)', '', phones)

    # 받침 이응 'ng'으로 처리 (Velar nasal in coda position)
    phones = re.sub('oh-', 'ng-', phones)
    phones = re.sub('oh([# ]|$)', 'ng', phones)

    # Remove all characters except Hangul and syllable delimiter (hyphen; '-')
    phones = re.sub('(\W+)\-', '\\1', phones)
    phones = re.sub('\W+$', '', phones)
    phones = re.sub('^\-', '', phones)
    return phones

def phone2prono(phones, rule_in, rule_out):
    # Apply g2p rules
    for pattern, replacement in zip(rule_in, rule_out):
        # print pattern
        phones = re.sub(pattern, replacement, phones)
        prono = phones
    return prono

def addPhoneBoundary(phones):
    # Add a comma (,) after every second alphabets to mark phone boundaries
    ipos = 0
    newphones = ''
    while ipos + 2 <= len(phones):
        if phones[ipos] == u'-':
            newphones = newphones + phones[ipos]
            ipos += 1
        elif phones[ipos] == u' ':
            ipos += 1
        elif phones[ipos] == u'#':
            newphones = newphones + phones[ipos]
            ipos += 1

        newphones = newphones + phones[ipos] + phones[ipos+1] + u','
        ipos += 2

    return newphones

def addSpace(phones):
    ipos = 0
    newphones = ''
    while ipos < len(phones):
        if ipos == 0:
            newphones = newphones + phones[ipos] + phones[ipos + 1]
        else:
            newphones = newphones + ' ' + phones[ipos] + phones[ipos + 1]
        ipos += 2

    return newphones

def graph2prono(graphs, rule_in, rule_out):

    romanized = graph2phone(graphs)
    romanized_bd = addPhoneBoundary(romanized)
    prono = phone2prono(romanized_bd, rule_in, rule_out)

    prono = re.sub(u',', u' ', prono)
    prono = re.sub(u' $', u'', prono)
    prono = re.sub(u'#', u'-', prono)
    prono = re.sub(u'-+', u'-', prono)

    prono_prev = prono
    identical = False
    loop_cnt = 1

    while not identical:
        prono_new = phone2prono(re.sub(u' ', u',', prono_prev + u','), rule_in, rule_out)
        prono_new = re.sub(u',', u' ', prono_new)
        prono_new = re.sub(u' $', u'', prono_new)

        if re.sub(u'-', u'', prono_prev) == re.sub(u'-', u'', prono_new):
            identical = True
            prono_new = re.sub(u'-', u'', prono_new)
        else:
            loop_cnt += 1
            prono_prev = prono_new

    return prono_new

class Phone(object):
    def __init__(self):
        self.ons = None
        self.nuc = None
        self.cod = None

    def to_list(self):
        phoneme_list = []
        for var in vars(self):
            value = getattr(self, var)
            if value is not None:
                phoneme_list.append(value)

        return phoneme_list

    def num(self):
        num = 0
        for var in vars(self):
            value = getattr(self, var)
            if value is not None:
                num += 1

        return num

def encode(graph, rulebook='./g2p/rulebook.txt'):
    [rule_in, rule_out] = readRules(rulebook)
        
    prono = graph2prono(graph, rule_in, rule_out)
    prono = prono.split(' ')
    encoded_prono = [Phone()] 

    for p in prono:
        if p in ONS:
            if ONS.index(p) < ONS.index('oh'):
                encoded_prono[-1].ons = ONS.index(p) + 1
            else:
                encoded_prono[-1].ons = ONS.index(p)
        elif p in NUC:
            encoded_prono[-1].nuc = NUC.index(p) + len(ONS)
            encoded_prono.append(Phone())
        elif p in COD:
            encoded_prono[-2].cod = RCD.index(p) + len(ONS) + len(NUC)

    return encoded_prono[:-1]

def decode(encoded_prono):
    prono = []
    for p in encoded_prono:
        phone = ''
        if p.ons is not None: 
            if p.ons < ONS.index('oh'):
                phone += ONS[p.ons-1]
            else:
                phone += ONS[p.ons]
        if p.nuc is not None:
            phone += NUC[p.nuc-len(ONS)]
        if p.cod is not None:
            phone += RCD[p.cod-(len(ONS)+len(NUC))]

        prono.append(phone)

    return prono