import openai

def summarize(prompt):
    augmented_prompt = f'Lue seuraava kansanedustajan eduskunnassa pitämä puhe:\n {prompt}\nTiivistä puheen sisältö 5-7 virkkeeseen. Kiinnitä erityistä huomiota tärkeimpiin aiheisiin ja siihen, miten puhuja niihin suhtatuu.'
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

# Iterate over all speeches and write API responses to file
def save_to_disk(speeches, summary_round):
    with open(f'results/summaries_pentti-tiusanen_2010_{summary_round}.txt', 'w') as f:
        for idx, speech in enumerate(speeches):
            f.write(f'### Summary {idx}\n')
            f.write(f'{speech}\n')
            
            
def get_key():
    # Get OpenAI authorization key
    with open('auth-key', 'r') as f:
        openai.api_key = f.read()

def get_data():
    # Get speeches in a list
    with open('../data/summarization_data/predared-data_pentti-tiusanen_2010.txt', 'r') as f:
        speeches = f.readlines()
    return speeches



get_key()
data = get_data()
data = data[0:10]
summaries = []
for speech in data:
    summaries.append(summarize(speech))
    
save_to_disk(summaries, 0)