import openai
import tiktoken
from tqdm import tqdm


def num_tokens_from_string(speech):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    num_tokens = len(encoding.encode(speech))
    return num_tokens


# merge short speeches and split long ones. This needs some work if working with another set of speeches with longer max speech length.
def combine_and_split(data):
    combined_and_split = []
    combined = []
    len_of_combined = []
    for idx, speech in enumerate(input_ids):
        len_of_speech = num_tokens_from_string(speech)
        if len_of_speech > 3000:
            start, end = speech[:len(speech)//2], speech[:len(speech)//2]
            combined_and_split.append(speech[len(speech)//2:])
            combined_and_split.append(speech[:len(speech)//2])
        elif len_of_speech < 2000:
            combined += speech
            len_of_combined += len_of_speech
        else:
            combined_and_split.append(speech)
        if len_of_combined > 2000 or idx+1 == len(input_ids):
            if len(combined) != 0:
                combined_and_split.append(combined)
                combined = []

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

