import openai
import tiktoken
from tqdm import tqdm
import textwrap


def num_tokens_from_string(speech):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    num_tokens = len(encoding.encode(speech))
    return num_tokens


# merge short speeches and split long ones. This needs some work if working with another set of speeches with longer max speech length.
def combine_and_split(data: list, max_len = 7500, min_len = 3000):
    """ Truncate speeches to max_len and combine them to min_len.
    First we use textwrapper to return a list of speeches, which are at most max_len long. Longer speeches than that are split.
    Then we iterate over that list. Speeches longer than min_len are just appended to final_speeches.
    Speeches shorter than min_len are combined together until min_len is reached and the added to final_speeches.
    In the end, if there is something left that does not reach min_len, it is still added to final_speeches.
    This function might return speeches that are longer than max_len in min_len is set too high,
    because the combination of too speeches shorter than min_len might be longer than max_len.
    """
    if min_len >= max_len:
        raise Exception('min_len cannot be higher than max_len!')
    split_speeches = []
    combined_speeches = ''
    final_speeches = []
    for speech in data:
        split_speeches += textwrap.wrap(speech, width=max_len)
    for idx, speech in enumerate(split_speeches):
        if len(speech) >= min_len:
            final_speeches.append(speech)
        else:
            combined_speeches += speech
            combined_speeches += ' ' # Make sure there's a space between speeches.
        if len(combined_speeches) > min_len or idx+1 == len(split_speeches):
            if len(combined_speeches) != 0:
                final_speeches.append(combined_speeches)
                combined_speeches = ''
    return final_speeches


def summarize(prompt, round_num):
    if round_num == 1: # On first round, use a different prompt
        augmented_prompt = f'Lue seuraava kansanedustajan eduskunnassa pitämä puhe:\n {prompt}\nTiivistä puheen sisältö 5-7 virkkeeseen. Kiinnitä erityistä huomiota tärkeimpiin aiheisiin ja siihen, miten puhuja niihin suhtatuu.'
    else:
        augmented_prompt = f'Lue seuraavat kuvaukset kansanedustajan eduskunnassa pitämistä puheista:\n {prompt}\nLaadi yksi 5-7 virkkeen tiivistelmä puheiden sisällöstä. Kiinnitä erityistä huomiota tärkeimpiin aiheisiin ja siihen, miten puhuja niihin suhtatuu.'
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "Olet avulias assistentti, joka kirjoittaa tiivistelmiä."},
            {"role": "user", "content": augmented_prompt}
        ],
        temperature = 0.2,
        max_tokens = 500, # max length of response
    )
    return response['choices'][0]['message']['content'] # return only response text and ignore all metadata

def return_prompt(prompt):
    '''
    This is just a test function to check that the code runs. Can be ignored.
    '''
    augmented_prompt = 'This is a summary.'
    return augmented_prompt

# Iterate over all speeches and write API responses to file
def save_to_disk(speeches, summary_round):
    with open(f'results/summaries_pentti-tiusanen_2010_{summary_round}.txt', 'w') as f:
        for idx, speech in enumerate(speeches):
            f.write(f'### Summary {idx}\n')
            f.write(f'{speech}\n')


# Combine summaries for another summarization round
def combine(summaries, num_to_combine):
    to_summarize = []
    count = 0
    combined = ''
    for idx, i in enumerate(summaries):
        count += 1
        combined += i
        if count == num_to_combine or idx+1 == len(summaries):
            if not len(combined) == 0:
                to_summarize.append(combined)
                count = 0
                combined = ''
    return to_summarize

def get_key():
    # Get OpenAI authorization key
    with open('auth-key', 'r') as f:
        openai.api_key = f.read()

def get_data():
    # Get speeches in a list
    with open('../data/summarization_data/prepared-data_pentti-tiusanen_2010.txt', 'r') as f:
        speeches = f.readlines()
    return speeches

def main()
    get_key() # load API key
    data = get_data() # load initial data
    round_num = 0
    go_on = True
    while go_on:
        if len(data) == 1: # if there is only one speech left, this is the last round
            go_on = False
        round_num += 1
        if round_num == 7: # If something goes wrong, stop after 7 rounds
            go_on = False
        summaries = []
        for speech in tqdm(data):
            summaries.append(summarize(speech, round_num)) # Do summarization
        save_to_disk(summaries, round_num) # Save summaries to disk
        data = combine(summaries, 8) # Combine summaries
        
if '__name__' == '__main__':
    main()

