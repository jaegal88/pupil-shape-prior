from __future__ import print_function
import time

def print_expected_time_train(total_batch, total_epoch, start_time, cost):
    s = "Expected time : %d hour %d min %d sec" % (((total_batch * total_epoch / 10 * cost) / 60) / 60,
                                                   (total_batch * total_epoch / 10 * cost) / 60 % 60,
                                                   (total_batch * total_epoch / 10 * cost) % 60)
    print(s)
    expected_end_time = time.localtime(start_time + (total_batch * total_epoch / 10 * cost))
    s = "\t\t%d.%d.%d. %02d:%02d:%02d" % (expected_end_time.tm_year, expected_end_time.tm_mon,
                                      expected_end_time.tm_mday, expected_end_time.tm_hour,
                                      expected_end_time.tm_min, expected_end_time.tm_sec)
    print(s)


def print_expected_time_test(numfile, start_time, cost):
    s = "Expected time : %d hour %d min %d sec" % (((numfile / 10 * cost) / 60) / 60,
                                                   (numfile / 10 * cost) / 60 % 60,
                                                   (numfile / 10 * cost) % 60)
    print(s)
    expected_end_time = time.localtime(start_time + (numfile / 10 * cost))
    s = "\t\t\t\t%d.%d.%d. %02d:%02d:%02d" % (expected_end_time.tm_year, expected_end_time.tm_mon,
                                        expected_end_time.tm_mday, expected_end_time.tm_hour,
                                        expected_end_time.tm_min, expected_end_time.tm_sec)
    print(s)