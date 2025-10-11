# WWWPS

world wide web positioning system.

Uses ngrams to weigh links against a search query and crawl towards higher rating hits.

Copied from the WWW page on [my digital garden](https://lukal.neocities.org/www.html). Text there might differ.

---

So the starting assumption here is that the internet is perfectly fine and that useful/interesting websites are still there - it's just that the discoverability is not that good. Well, why not try and solve discoverability with a positioning system? If you think of search engines as maps that you use to navigate the web, think of WWWPS as a GPS which knows in which direction something is.

The user interface for a world wide web positioning system? A plaintext messageboard.

This is an example of [immortal software](index.html#immortal%3F) - any moderately skilled programmer on the planet can easily build the following two things in a relatively short amount of time:

- a rudimentary messageboard with enumerated posts, 1 level replies, quoting, and pagination
- a respectful crawler (waits for 10-30 seconds between each HTTP request) bundled together with a list of like top 1000 most common words and top 5-10 TLDs for starting points, an ngram (trigram) similarity comparison, and a gradually decreasing threshold for next most similar page to look at.

So you can make a post on the messageboard like

```
blah blah blah yadda yadda and Python makes sense as one of the world's most popular languages at the time of writing.

lookup: search engine development python
```

which the crawler picks up and as it finds increasingly better matches it replies to your post with relevant pages it finds. You could even micromanage it live by replying to yourself

```
ok let's narrow it down a bit

lookup: search engine development python
update: search engine development python, not tutorial
```

(a negative (see "not tutorial" part in the previous example) part of the query has its scoring inverted (think in terms of multiplication with -1)) or tell it to stop by doing

```
ok,thanks

halt: search engine development python
```

and it will stop. The syntax is just an example and obviously you need to set some kind of secret/password in order to instruct the crawler to change course or do something, but yeah that's essentially all there is to it.

One downside is that it's a bit slower at first and you might get a reply from a human being with useful results in the meantime but that's why it's a message board. You can discuss and collaborate on querying methods among other things and discuss the results with other people.

You would obviously be able to make posts hidden/private/password-protected.

Note that the speed does improve with each new query because they are compared against older queries and better scored matches are reused as starting points (this is already implemented) plus horizontal scaling could speed up crawling (not implemented yet) because the on-disk cache (cache is implemented) can be shared (sharing is not implemented).

My WWWPS crawler implementation is here https://github.com/lukal-x/wwwps I got the idea by experimenting in that repository.

I'll give it a messageboard interface ASAP and put it up on http://searchboard.ftp.sh (https://searchboard.ftp.sh).