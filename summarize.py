import openai

# Get OpenAI authorization key
with open('auth-key', 'r') as f:
    openai.api_key = f.read()

# Get speeches in a list
with open('../data/summarization_data/predared-data_pentti-tiusanen_2010.txt', 'r') as f:
    speeches = f.readlines()

def summarize(prompt):
    augmented_prompt = f'Lue seuraava kansanedustajan eduskunnassa pitämä puhe:\n {prompt}\nTiivistä puheen sisältö 5-7 virkkeeseen. Kiinnitä erityistä huomiota tärkeimpiin aiheisiin ja siihen, miten puhuja niihin suhtatuu.'
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        prompt=[
            {"role": "system", "content": "Olet avulias assistentti, joka kirjoittaa tiivistelmiä."},
            {"role": "user", "content": augmented_prompt}
        ],
        temperature = 0.2,
        max_tokens = 500, # max length of response
    )
    return response['choices'][0]['message']['content'] # return only response text and ignore all metadata

# Iterate over all speeches
summaries = []
for speech in speeches:
    summaries.append(summarize(speech))

# Save summaries to disk
with open('results/summaries_pentti-tiusanen_2010.txt', 'w') as f:
    for summary in summaries:
        f.write(summary)
        f.write('\n')
