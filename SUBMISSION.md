#Week 1 Submission of GenAI Task: Multi-Turn Terminal Chatbot

What I learn-
I start from basics and I learn about LLM ,API .How AI works.What is the general format of API calling .I learn how AI remembers the previous conversations by sending all messages list to model.What we can extract from response.What we can see from response.choices and response.usage


How I Implemeted the Task-
I learn the basic things and general format like sending the prompt to model then
I started by creating a basic API call using the OpenAI SDK with OpenRouter's base URL.
After understanding the single API call, I converted it into a multi-turn chatbot by using a while True loop. This loop keeps asking the user for input until the user types exit or quit.
I practice build 1 and 2 ,see the responce.choices and response.usage
Then for the final submission I implemented a ChatAgent class.The class stores the model name, system prompt, message history, token usage, and OpenRouter client.
    Why- So that every agent has its own prompt ,own history.
I made the model like that it remembers the previous conversation for every message, I append the user's message to the messages list, send the full messages list to the model, extract the assistant's reply from the response object, append the assistant reply to list.    
    Why - Because it is very important to remember the previous conversations to make work better.
Then I also keep a rolling buffer that keeps only the last 7 turns. 
    Why - Because when model remember list get too long due to which model becomes slower.