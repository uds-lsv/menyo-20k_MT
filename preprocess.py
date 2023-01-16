import os
import pandas as pd

from nltk.corpus import words
import preprocessing

def standardize_to_NFC(yo_sent):
    with open('temp/yoruba_nonNFC.txt', 'w') as f:
        #print(yo_sent)
        for i, yor in enumerate(yo_sent):
            try:
                f.write(yor + '\n')
            except:
                print('error', yo_sent[i-1])
                print('error', i)
                print('error', yo_sent[i + 1])

            #    f.write('REMOVE SENTENCE' + '\n')
    preprocessing.normalize_diacritics_file('temp/yoruba_nonNFC.txt', 'temp/yorubaNFC.txt')

    with open('temp/yorubaNFC.txt') as f:
        all_lines = f.readlines()

    all_lines  = [sent.strip() for sent in all_lines]

    return all_lines

def filter_single_word_sentence(eng_sents, yor_sents):

    eng_inds = set([s for s, sent in enumerate(eng_sents) if len(sent.split()) > 1])
    yor_inds = set([s for s, sent in enumerate(yor_sents) if len(sent.split()) > 1])
    common_inds = sorted(list(eng_inds & yor_inds))

    eng_texts = [eng_sents[s].strip() for s in common_inds]
    yor_texts = [yor_sents[s].strip() for s in common_inds]

    return eng_texts, yor_texts


def process_dev_test(input_dir):
    output_dir = 'data/'
    file_sents = {}
    input_dev_test = {'yoruba_proverbs.csv':(250,250), 'ted_0_937.csv':(438, 500),
                   'out_of_his_mind_corrected.csv':(1008, 1009),
                 'global_voices.csv':(1392, 1392), 'Kolibri_part2_omo_yoruba_digital.csv':(312,273)}

    map_domain_dict = {'yoruba_proverbs.csv':'proverbs', 'ted_0_937.csv':'ted_first937',
                   'out_of_his_mind_corrected.csv': 'book',
                 'global_voices.csv':'gv_news', 'Kolibri_part2_omo_yoruba_digital.csv':'digital'}

    for file in input_dev_test:
        print(file)
        sum_dev_test = input_dev_test[file][0] + input_dev_test[file][1]
        df = pd.read_csv(input_dir+file, names=['English','Yoruba'])
        #print(df.columns)
        #print(df.head())
        df.columns = ['en', 'yo']
        en_sents = list(df['en'].values)[1:]
        yo_sents = list(df['yo'].values)[1:]
        en_sents = en_sents[:sum_dev_test]
        yo_sents = yo_sents[:sum_dev_test]

        yo_sents = standardize_to_NFC(yo_sents)

        file_sents[file] = (en_sents, yo_sents)

    merged_dev_en = []
    merged_dev_yo = []
    merged_test_en = []
    merged_test_yo = []
    for file in file_sents:
        dev_sents_en = file_sents[file][0][:input_dev_test[file][0]]
        dev_sents_yo = file_sents[file][1][:input_dev_test[file][0]]

        test_sents_en = file_sents[file][0][input_dev_test[file][0]:]
        test_sents_yo = file_sents[file][1][input_dev_test[file][0]:]

        dev_sents_en, dev_sents_yo = filter_single_word_sentence(dev_sents_en, dev_sents_yo)
        test_sents_en, test_sents_yo = filter_single_word_sentence(test_sents_en, test_sents_yo)
        print(file, 'DEV', 'TEST', len(dev_sents_en), len(test_sents_en))

        merged_dev_en += dev_sents_en
        merged_dev_yo += dev_sents_yo
        merged_test_en += test_sents_en
        merged_test_yo += test_sents_yo

        t_df = pd.DataFrame(test_sents_en)
        t_df.columns = ['english']
        t_df['yoruba'] = test_sents_yo
        t_df.to_csv(output_dir + 'test_'+map_domain_dict[file]+'.tsv', sep='\t', header=True, index=False)


    dev_df = pd.DataFrame(merged_dev_en)
    dev_df.columns = ['english']
    dev_df['yoruba'] = merged_dev_yo
    dev_df.to_csv(output_dir+'dev.tsv', sep='\t', header=True, index=False)

    print('# DEV set', dev_df.shape)

    '''
    test_df = pd.DataFrame(merged_test_en)
    test_df.columns = ['english']
    test_df['yoruba'] = merged_test_yo
    test_df.to_csv(output_dir+'test.tsv', sep='\t', header=True, index=False)
    '''

    #print('# TEST set', test_df.shape)


def two_parallel_texts_to_csv(dir_name, en_file, yo_file, new_file_name):
    with open(dir_name+en_file) as f, open(dir_name+yo_file) as f2:
        en_text = f.read()
        yo_text = f2.read()

    en_sentences = en_text.splitlines()
    yo_sentences = yo_text.splitlines()
    eng_texts, yor_texts = filter_single_word_sentence(en_sentences, yo_sentences)

    yor_texts = standardize_to_NFC(yor_texts)

    df = pd.DataFrame(eng_texts)
    df.columns = ['English']
    df['Yoruba'] = yor_texts
    df.to_csv('verified/'+new_file_name, sep=',', header=True, index=False)

    print(dir_name+new_file_name+" # sentences", df.shape)
    return df

def combine_all_test_sets(dir_name):

    # Ted talks
    df_ted500 = pd.read_csv('data/test_ted_first937.tsv', sep='\t')
    df_ted = pd.read_csv(dir_name + 'tedTalks_938_2945.csv')
    df_ted.columns = ['english', 'yoruba']
    df_ted_test = pd.concat([df_ted500, df_ted.iloc[0:1500]])
    #df_ted_test = df_ted.iloc[0:1500]
    df_ted_test.to_csv('data/test_tedTalks.tsv', sep='\t', header=True, index=False)
    print('Ted talks, test: ', df_ted_test.shape)


    df_gv = pd.read_csv('data/test_gv_news.tsv', sep='\t')
    df_gv.columns = ['English', 'Yoruba']
    # VON news
    df_von1 = pd.read_csv(dir_name + 'von_part1.csv')
    df_von2 = pd.read_csv(dir_name + 'von_part2.csv')
    df_von3 = pd.read_csv(dir_name + 'von_part3.csv')
    df_von4 = pd.read_csv(dir_name + 'von_part4.csv')

    df_news = pd.concat([df_gv, df_von1, df_von2, df_von3, df_von4])
    df_news.columns = ['english', 'yoruba']
    df_news.to_csv('data/test_news.tsv', sep='\t', header=True, index=False)
    print('News, test: ', df_news.shape)


def combine_all_train_data(dir_name):
    df1 = pd.read_csv(dir_name + 'cc_1.csv')
    print('CC: ', df1.shape)
    df2 = pd.read_csv(dir_name + 'Engine_room.csv')
    print('Engine Room: ', df2.shape)
    df3 = pd.read_csv(dir_name + 'global_voices_new.csv')
    print('GV news: ', df3.shape)
    df4 = pd.read_csv(dir_name + 'radio_transcripts.csv')
    print('Radio news: ', df4.shape)
    df5 = pd.read_csv(dir_name + 'kolibri_fixed.csv')
    print('Kolibri Tech: ', df5.shape)
    df6 = pd.read_csv(dir_name + 'omoYoruba_corrected.csv', usecols=[1,2])
    print('Various: ', df6.shape)
    df7 = pd.read_csv(dir_name + 'Unsane_movie_transcript_corrected.csv', usecols=[1,2])
    print('Movie transcript: ', df7.shape)
    df8 = pd.read_csv(dir_name + 'von_part5.csv')
    print('VON news: ', df8.shape)
    df9 = pd.read_csv(dir_name + 'von_part6.csv')
    print('VON news: ', df9.shape)
    df10 = pd.read_csv(dir_name + 'von_part7.csv')
    print('VON news: ', df10.shape)
    df_prob = pd.read_csv(dir_name + 'yoruba_proverbs.csv', usecols=[1, 2])
    df11 = df_prob.iloc[500:]
    print('Proverbs: ', df11.shape)
    df12 = pd.read_csv(dir_name + 'uhdr.csv')
    print('UHDR: ', df12.shape)
    df_ted = pd.read_csv(dir_name + 'tedTalks_938_2945.csv')
    df13 = df_ted.iloc[1500:]
    print('Ted talks: ', df13.shape)
    df14 = pd.read_csv(dir_name + 'jw_news.csv')
    print('JW news: ', df14.shape)

    df_all = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14])
    eng_sents = df_all['English'].values
    yor_sents = df_all['Yoruba'].values
    for y, yo in enumerate(eng_sents):
        if len(yo.split()) < 2:
            print(y, yo)
    eng_texts, yor_texts = filter_single_word_sentence(eng_sents, yor_sents)
    yor_texts = standardize_to_NFC(yor_texts)

    df = pd.DataFrame(eng_texts)
    df.columns = ['English']
    df['Yoruba'] = yor_texts
    df.to_csv('data/train.tsv', sep='\t', header=True, index=False)
    print('Training size: ', df.shape)


if __name__ == '__main__':
    #process_dev_test("verified/")
    dir_name = "verified/"
    '''
    df_cc1 = two_parallel_texts_to_csv(dir_name, "cc_1_en.txt", "cc_1_yo.txt", "cc_1.csv")
    df_engine = two_parallel_texts_to_csv(dir_name, "Engine_room_en.txt", "Engine_room_yo.txt", "Engine_room.csv")
    df_gv = two_parallel_texts_to_csv(dir_name, "global_voices_new_en.txt", "global_voices_new_yo.txt", "global_voices_new.csv")
    df_radio = two_parallel_texts_to_csv(dir_name, "radio_transcripts_en.txt", "radio_transcripts_yo.txt",
                                      "radio_transcripts.csv")
    df_ted = two_parallel_texts_to_csv(dir_name, "tedTalks_938_2945_en.txt", "tedTalks_938_2945_yo.txt",
                                         "tedTalks_938_2945.csv")
    df_von1 = two_parallel_texts_to_csv(dir_name, "von_part1_en.txt", "von_part1_yo.txt",
                                       "von_part1.csv")
    df_von2 = two_parallel_texts_to_csv(dir_name, "von_part2_en.txt", "von_part2_yo.txt",
                                       "von_part2.csv")
    df_von3 = two_parallel_texts_to_csv(dir_name, "von_part3_en.txt", "von_part3_yo.txt",
                                       "von_part3.csv")
    df_von4 = two_parallel_texts_to_csv(dir_name, "von_part4_en.txt", "von_part4_yo.txt",
                                       "von_part4.csv")
    df_von5 = two_parallel_texts_to_csv(dir_name, "von_part5_en.txt", "von_part5_yo.txt",
                                        "von_part5.csv")
    df_von6 = two_parallel_texts_to_csv(dir_name, "von_part6_en.txt", "von_part6_yo.txt",
                                        "von_part6.csv")
    df_von7 = two_parallel_texts_to_csv(dir_name, "von_part7_en.txt", "von_part7_yo.txt",
                                        "von_part7.csv")
    df_uhdr = two_parallel_texts_to_csv(dir_name, "uhdr_en.txt", "uhdr_yo.txt",
                                        "uhdr.csv")
    '''
    combine_all_test_sets(dir_name)
    #combine_all_train_data(dir_name)



