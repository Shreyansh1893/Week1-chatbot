# What I built and how it works

For this project I built a research assistant that runs in the terminal, similar to a basic version of Perplexity.

The user enters a question and the AI decides what information it needs to answer it. It can search the web using Serper, fetch the contents of webpages, and search academic papers through AlphaXiv. My code runs whichever tools the model asks for and sends the results back to it. This process repeats until the model has gathered enough information to answer the question.

I first got everything working from the command line so it was easier to debug. After that I added a Textual interface so the user could interact with it through a cleaner terminal UI.

The final result is a tool that can search multiple sources, read webpages and papers, and generate an answer with citations.

# One design decision and why

One decision I made was that to limit how much text is read from each webpage.

When I first started testing, some webpages were too long and sending the entire page to the model used a lot of tokens. Most of that content wasn't actually useful for answering the question. To avoid running into context limits, I only send part of the page content to the model.

The downside is that sometimes useful information could be further down the page, but in practice it worked well and allowed the agent to search more sources.

# Something that surprised me or didn't work as expected

The AlphaXiv integration was more complicated than I expected. I initially thought I could call it like a normal API, but it actually required an OAuth login flow through the browser and a separate helper process. It took some time to understand how all the pieces fit together.

I also ran into an annoying issue where my Serper API key kept showing up as missing even though it was present in my .env file. After debugging it, I found that one of the files was trying to read the environment variable before the .env file had been loaded.

Another issue came up during testing. One of my research queries triggered a lot of searches and page fetches, which made the conversation history very large. Eventually I hit OpenRouter's token limit, and shortly after that my free credits were exhausted, so the request couldn't finish.

# What I'd improve with more time

If I had more time, I would add streaming responses so users could see the answer being generated instead of waiting for the entire response.

I'd also improve the UI by showing tool activity in real time, such as searches, page fetches, and paper lookups. That would make it easier to understand what the agent is doing.

Finally, I would add a way to summarize or remove older tool results from the conversation history. That would help keep the context size under control and reduce the chances of hitting token limits during longer research sessions.