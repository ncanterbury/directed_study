



def process_data(text_file, output_file):
    with open(output_file, 'w') as f2:
        with open(text_file) as f:
            data = f.readlines()

            for line in data:
                words = line.split(";@;")
                print words[-1] 
                f2.write(words[-1])


def main():
    text_file = "test_data_awk.txt"
    output_file = "test_getting_tweet.txt"
    process_data(text_file, output_file)

if __name__ == ("__main__"):
    main()
