from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from itertools import izip 

import datetime 

import matplotlib.dates
import csv

def make_prediction(d_array, p_array):

    final_dates = []
    obama_prob_array = []
    romney_prob_array = []

    #prev_day = datetime.datetime.strptime(d_array[0][0] + d_array[0][1]+d_array[0][2], "%m%d%Y")
    prev_day = d_array[0][2]+"-"+d_array[0][1]+"-"+d_array[0][0]



    ini_pred = p_array[0]
    
    cnt_obama_pos = 1
    cnt_obama_neg = 1
    cnt_romney_pos = 1
    cnt_romney_neg = 1

    for index in range(1, len(d_array)):
        try:
            int(d_array[index][2])
            curr_day = d_array[index][2]+"-"+d_array[index][1]+"-"+d_array[index][0]


            if curr_day > prev_day:
                # make calculation
                obama_pos_neg = (cnt_obama_pos / cnt_obama_neg)
                romney_pos_neg = (cnt_romney_pos / cnt_romney_neg)
                prob_obama = (obama_pos_neg / (obama_pos_neg + romney_pos_neg))
                prob_romney = (romney_pos_neg / (romney_pos_neg + obama_pos_neg))
                
                print cnt_obama_pos
                print cnt_obama_neg

                print ("This is probability obama:", prob_obama)
                print ("This is probability romney:", prob_romney)
                print ("This is date:", prev_day)
                
                # add to arrays
                final_dates.append(prev_day)
                obama_prob_array.append(prob_obama)
                romney_prob_array.append(prob_romney)

                # reset counters to 1
                cnt_obama_pos = 1
                cnt_obama_neg = 1
                cnt_romney_pos = 1
                cnt_romney_neg = 1
                prev_day = curr_day

            
            elif curr_day == prev_day:
                if p_array[index] == 'pro_obama':
                    cnt_obama_pos += 1
                    cnt_romney_neg += 1
                elif p_array[index] == 'pro_romney':
                    cnt_romney_pos += 1
                    cnt_obama_neg += 1
                prev_day = curr_day


        except ValueError:
            prev_day = curr_day
            pass
        
    print final_dates
    print obama_prob_array
    print romney_prob_array

    date_objects = [datetime.datetime.strptime(date, '%Y-%d-%m').date() for date in final_dates]

    plt.plot(date_objects, obama_prob_array)
    

def make_data(predictions, election_data, output):
    
    open_election_data = open(election_data)
    election_reader = csv.reader(open_election_data)
    open_prediction = open(predictions)
    prection_reader = csv.reader(open_prediction)
    open_output = open(output, 'w')
    
    date_array = []
    predict_array = []

    line = 0
    for line_elec, line_predict in izip(election_reader, prection_reader):
        if line > 10000:
            break
        #print "This is line"

        #print line
        line+=1
        if line % 100000 == 0:
            print line 
        #print "check elec"
        day = line_elec[0][8:10]
        month = line_elec[0][5:7]
        year = line_elec[0][0:4]
        date = [str(month), str(day),str(year)]
        #print ("Day/Month/Year", date) 
        date_array.append(date)

        #print "check predict"
        predict = line_predict[2][2:]
        predict_array.append(predict)

    print "is date length"
    print len(date_array)
    print len(predict_array)
    print date_array
    print predict_array
    open_output.write(str(date_array))
    open_election_data.close()
    open_prediction.close()
    open_output.close()


    make_prediction(date_array, predict_array)


def main():
    predictions = "arff_files/predictions.csv"
    election_data = "tweets/cut_down_new_election_data.txt"
    prediction_with_time = "tweets/prediction_with_time.txt"
    make_data(predictions, election_data, prediction_with_time)


if __name__ =="__main__":
    main()
