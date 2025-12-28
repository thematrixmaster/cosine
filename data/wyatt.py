import os
from pathlib import Path
from Bio.Seq import Seq
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from evo.antibody import parallel_align_sequences

tqdm.pandas()


def save_peint_df_to_txt(peint_df, base_path):
    parent_heavy_col = 'parent_heavy_aa_aho'
    parent_light_col = 'parent_light_aa_aho'
    child_heavy_col = 'child_heavy_aa_aho'
    child_light_col = 'child_light_aa_aho'
    folder_name = "aho"

    for donor in tqdm(pd.unique(peint_df.sample_id)):
        subset = peint_df[peint_df.sample_id == donor]

        parent = subset[parent_heavy_col] + "." + subset[parent_light_col]
        child = subset[child_heavy_col] + "." + subset[child_light_col]
        lines = parent + " " + child + " " + subset.branch_length.astype(str)

        save_dir = base_path + folder_name + "/"
        os.makedirs(save_dir, exist_ok=True)

        outfile = save_dir + "{0}.txt".format(donor)
        with open(outfile, "w") as f:
            f.write("{0} transitions\n".format(subset.shape[0]))
            f.write("\n".join(lines))

        parent = subset[parent_heavy_col]
        child = subset[child_heavy_col]
        lines = parent + " " + child + " " + subset.branch_length.astype(str)

        # save_dir = save_dir = base_path + "edges_heavy/" + folder_name + "/"
        # os.makedirs(save_dir, exist_ok=True)

        # outfile = save_dir + "{0}.txt".format(donor)
        # with open(outfile, "w") as f:
        #     f.write("{0} transitions\n".format(subset.shape[0]))
        #     f.write("\n".join(lines))

        # parent = subset[parent_light_col]
        # child = subset[child_light_col]
        # lines = parent + " " + child + " " + subset.branch_length.astype(str)

        # save_dir = base_path + "edges_light/" + folder_name + "/"
        # os.makedirs(save_dir, exist_ok=True)

        # outfile = save_dir + "{0}.txt".format(donor)
        # with open(outfile, "w") as f:
        #     f.write("{0} transitions\n".format(subset.shape[0]))
        #     f.write("\n".join(lines))
        
        
def save_peint_df_to_csv(peint_df, csv_path):
    train_data = []
    test_data = []  
    for donor in tqdm(pd.unique(peint_df.sample_id)):
        subset = peint_df[peint_df.sample_id == donor]
        parent_heavy_aho_seqs = subset['parent_heavy_aa_aho'].tolist()
        parent_light_aho_seqs = subset['parent_light_aa_aho'].tolist()
        child_heavy_aho_seqs = subset['child_heavy_aa_aho'].tolist()
        child_light_aho_seqs = subset['child_light_aa_aho'].tolist()
        parent_aho_tuples = list(zip(parent_heavy_aho_seqs, parent_light_aho_seqs))
        child_aho_tuples = list(zip(child_heavy_aho_seqs, child_light_aho_seqs))
        unique_aho_tuples = list(set(parent_aho_tuples + child_aho_tuples))
        if donor == 'd4':
            test_data.extend(unique_aho_tuples)
        else:
            train_data.extend(unique_aho_tuples)
    # put donor 4 sequences into the test set
    test_df = pd.DataFrame(test_data, columns=['fv_heavy_aho', 'fv_light_aho'])
    test_df['partition'] = 'test'
    # split the remaining sequences into train and validation sets (5% val set)
    train_df = pd.DataFrame(train_data, columns=['fv_heavy_aho', 'fv_light_aho'])
    train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=42)    
    train_df['partition'] = 'train'
    val_df['partition'] = 'val'
    # concatenate the train, validation, and test sets
    all_df = pd.concat([train_df, val_df, test_df])
    all_df.to_csv(csv_path, index=False)


def main():
    peint_df = pd.read_csv("/accounts/projects/yss/stephen.lu/peint-workspace/main/data/wyatt/subs/peint_df.csv.gz", compression='gzip')

    # # give each sequence a unique id
    # peint_df['parent_heavy_aa_uid'] = peint_df['family'] + ';' + peint_df['parent_name'] + ';' + 'heavy'
    # peint_df['parent_light_aa_uid'] = peint_df['family'] + ';' + peint_df['parent_name'] + ';' + 'light'
    # peint_df['child_heavy_aa_uid'] = peint_df['family'] + ';' + peint_df['child_name'] + ';' + 'heavy'
    # peint_df['child_light_aa_uid'] = peint_df['family'] + ';' + peint_df['child_name'] + ';' + 'light'

    all_parent_heavy_aa = peint_df['parent_heavy_aa'].progress_apply(lambda x: ('H_chain_1', x)).tolist()
    all_parent_light_aa = peint_df['parent_light_aa'].progress_apply(lambda x: ('L_chain_1', x)).tolist()
    all_child_heavy_aa = peint_df['child_heavy_aa'].progress_apply(lambda x: ('H_chain_1', x)).tolist()
    all_child_light_aa = peint_df['child_light_aa'].progress_apply(lambda x: ('L_chain_1', x)).tolist()
    
    all_parent_heavy_aa_aho = parallel_align_sequences(all_parent_heavy_aa, n_jobs=28)
    all_parent_light_aa_aho = parallel_align_sequences(all_parent_light_aa, n_jobs=28)
    all_child_heavy_aa_aho = parallel_align_sequences(all_child_heavy_aa, n_jobs=28)
    all_child_light_aa_aho = parallel_align_sequences(all_child_light_aa, n_jobs=28)

    # all_heavy_aho_seqs = list(set([x[1] for x in all_parent_heavy_aa_aho + all_child_heavy_aa_aho]))
    # all_light_aho_seqs = list(set([x[1] for x in all_parent_light_aa_aho + all_child_light_aa_aho]))
    # print(f"Number of unique heavy AHo sequences: {len(all_heavy_aho_seqs)}")
    # print(f"Number of unique light AHo sequences: {len(all_light_aho_seqs)}")
    
    all_parent_heavy_aa_aho = [x[1] for x in all_parent_heavy_aa_aho]
    all_child_heavy_aa_aho = [x[1] for x in all_child_heavy_aa_aho]
    all_parent_light_aa_aho = [x[1] for x in all_parent_light_aa_aho]
    all_child_light_aa_aho = [x[1] for x in all_child_light_aa_aho]
    
    heavy_aho_seq_lengths = [len(x) for x in all_parent_heavy_aa_aho + all_child_heavy_aa_aho]
    light_aho_seq_lengths = [len(x) for x in all_parent_light_aa_aho + all_child_light_aa_aho]
    print(f"heavy AHo sequence length: {set(heavy_aho_seq_lengths)}")
    print(f"light AHo sequence length: {set(light_aho_seq_lengths)}")
    breakpoint()

    # check if the last character of the light chain aho is always a gap
    is_last_char_gap = [x[-1] == '-' for x in all_parent_light_aa_aho + all_child_light_aa_aho]
    print(f"Number of light chains with last character as gap: {sum(is_last_char_gap)}")
    print(f"Number of light chains with last character as not gap: {len(is_last_char_gap) - sum(is_last_char_gap)}")
    breakpoint()
    
    # slice off the last character of the light chain aho
    all_parent_light_aa_aho = [x[:-1] for x in all_parent_light_aa_aho]
    all_child_light_aa_aho = [x[:-1] for x in all_child_light_aa_aho]

    peint_df['parent_heavy_aa_aho'] = all_parent_heavy_aa_aho
    peint_df['parent_light_aa_aho'] = all_parent_light_aa_aho
    peint_df['child_heavy_aa_aho'] = all_child_heavy_aa_aho
    peint_df['child_light_aa_aho'] = all_child_light_aa_aho

    save_peint_df_to_txt(peint_df, base_path='/accounts/projects/yss/stephen.lu/peint-workspace/main/data/wyatt')
    save_peint_df_to_csv(peint_df, csv_path='/accounts/projects/yss/stephen.lu/peint-workspace/main/data/wyattaho/dwjs_wyatt.csv')
    breakpoint()
    
if __name__ == "__main__":
    main()
