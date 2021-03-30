import os

if __name__ == '__main__':
    os.makedirs("../user_data/official_data", exist_ok=True)
    lines = set()
    with open("../user_data/official_data/train.tsv", 'w') as fout:
        for input_file in [
            "/tcdata/gaiic_track3_round1_train_20210228.tsv",
            "/tcdata/gaiic_track3_round2_train_20210407.tsv",
        ]:
            for line in open(input_file):
                # if line in lines:
                #     continue
                # lines.add(line)
                line = line.strip()
                fout.write(line + '\n')
    with open("../user_data/official_data/testB.tsv", "w") as fout:
        for input_file in [
            "/tcdata/gaiic_track3_round1_testA_20210228.tsv",
            "/tcdata/gaiic_track3_round1_testB_20210317.tsv"
        ]:
            for line in open(input_file):
                line = line.strip()
                fout.write(line + '\n')
