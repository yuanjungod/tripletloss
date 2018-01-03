# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

import os
import codecs


class SampleData(object):
    global sample_person
    global _sample_negative
    global sample
    global _sample_label

    def __init__(self):
        self.sample_person = {}
        self._sample_negative = {}
        self.sample = []
        self._sample_label = {}
        lines = open('../data/train_val.txt', 'r')
        for line in lines:
            personname = line.split('@')[0]
            picname = line.split(' ')[0]
            self.sample.append(picname)
            if personname in self.sample_person.keys():
                self.sample_person[personname].append(picname)
            else:
                self.sample_person[personname] = []
                self.sample_person[personname].append(picname)
            self._sample_label[personname] = int(line.split(' ')[1])
        print(len(self.sample_person))


if __name__ == '__main__':
    sample = SampleData()
    # print sample._sample
