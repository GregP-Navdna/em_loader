import OpenAIApi from 'openai';
import { getKey, hasKey } from '../utils/keys.js';
import { strictFormat } from '../utils/text.js';

export class GPT {
    constructor(model_name, url, params) {
        this.model_name = model_name;
        this.params = params;

        let config = {};
        if (url)
            config.baseURL = url;

        if (hasKey('OPENAI_ORG_ID'))
            config.organization = getKey('OPENAI_ORG_ID');

        config.apiKey = getKey('OPENAI_API_KEY');

        this.openai = new OpenAIApi(config);
    }

    async sendRequest(turns, systemMessage, stop_seq='***') {
        let messages = [{'role': 'system', 'content': systemMessage}].concat(turns);

        // Check if this is a GPT-5 model
        const isGPT5 = this.model_name && (this.model_name.includes('gpt-5') || this.model_name === 'gpt-5-mini');
        
        let res = null;

        try {
            console.log('Awaiting openai api response from model', this.model_name);
            
            if (isGPT5) {
                // GPT-5 still uses chat.completions API but doesn't support 'stop' parameter
                const gpt5Pack = {
                    model: this.model_name,
                    messages,
                    ...(this.params || {})
                };
                
                // Remove unsupported parameters for GPT-5
                delete gpt5Pack.stop;
                
                // Add verbosity if specified
                if (this.params?.verbosity) {
                    gpt5Pack.text = {
                        verbosity: this.params.verbosity
                    };
                }
                
                console.log('Sending GPT-5 request with params:', JSON.stringify({
                    model: gpt5Pack.model,
                    message_count: messages.length,
                    has_verbosity: !!gpt5Pack.text?.verbosity
                }));
                
                let completion = await this.openai.chat.completions.create(gpt5Pack);
                
                console.log('GPT-5 response structure:', JSON.stringify({
                    has_choices: !!completion.choices,
                    choices_length: completion.choices?.length,
                    first_choice_keys: completion.choices?.[0] ? Object.keys(completion.choices[0]) : null
                }));
                
                if (completion.choices[0].finish_reason == 'length')
                    throw new Error('Context length exceeded');
                    
                console.log('Received GPT-5 response')
                res = completion.choices[0].message.content;
                
                if (!res) {
                    console.warn('GPT-5 returned empty response, full response:', JSON.stringify(completion));
                }
            } else {
                // Use traditional chat.completions API for non-GPT-5 models
                const pack = {
                    model: this.model_name || "gpt-3.5-turbo",
                    messages,
                    stop: stop_seq,
                    ...(this.params || {})
                };
                
                if (this.model_name.includes('o1') || this.model_name.includes('o3')) {
                    pack.messages = strictFormat(messages);
                    delete pack.stop;
                }
                
                console.log('Awaiting openai api response from model', this.model_name);
                let completion = await this.openai.chat.completions.create(pack);
                
                if (completion.choices[0].finish_reason == 'length')
                    throw new Error('Context length exceeded'); 
                    
                console.log('Received.')
                res = completion.choices[0].message.content;
            }
        }
        catch (err) {
            if ((err.message == 'Context length exceeded' || err.code == 'context_length_exceeded') && turns.length > 1) {
                console.log('Context length exceeded, trying again with shorter context.');
                return await this.sendRequest(turns.slice(1), systemMessage, stop_seq);
            } else if (err.message && err.message.includes('image_url')) {
                console.log(err);
                res = 'Vision is only supported by certain models.';
            } else {
                console.log(err);
                res = 'My brain disconnected, try again.';
            }
        }
        return res;
    }

    async sendVisionRequest(messages, systemMessage, imageBuffer) {
        const imageMessages = [...messages];
        imageMessages.push({
            role: "user",
            content: [
                { type: "text", text: systemMessage },
                {
                    type: "image_url",
                    image_url: {
                        url: `data:image/jpeg;base64,${imageBuffer.toString('base64')}`
                    }
                }
            ]
        });
        
        return this.sendRequest(imageMessages, systemMessage);
    }

    async embed(text) {
        if (text.length > 8191)
            text = text.slice(0, 8191);
        const embedding = await this.openai.embeddings.create({
            model: this.model_name || "text-embedding-3-small",
            input: text,
            encoding_format: "float",
        });
        return embedding.data[0].embedding;
    }

}



