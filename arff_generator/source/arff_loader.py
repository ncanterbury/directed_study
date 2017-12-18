import arff
import pandas as pd 

def arff_loader(csv_file):
    data_frame = pd.read_csv(csv_file)
    print data_frame
   



def main():
    csv_file = "~/directedstudyfall17/arff_files/csv_time.csv"
    arff_loader(csv_file)

if __name__ == "__main__":
    main()
