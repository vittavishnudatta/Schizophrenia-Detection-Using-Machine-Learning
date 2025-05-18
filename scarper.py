import praw
import pandas as pd

reddit_read_only = praw.Reddit(client_id="**********",		 # your client id
							client_secret="*********",	 # your client secret
							user_agent="********")	 # your user agent


subreddit = reddit_read_only.subreddit("teaching")
 
'''for post in subreddit.hot(limit=5000):
    df = post.elftext
    #df.to_csv('data.csv')
    print(type(df))
    print()'''

posts = subreddit.hot(limit=2500000)

posts_dict = {"Title": [], "Post Text": [],
			"ID": [], "Score": [],
			"Total Comments": [], "Post URL": [], 
            "Label" : []
			}

for post in posts:
    
    posts_dict["Title"].append(post.title)
    posts_dict["Post Text"].append(post.selftext)
    posts_dict["ID"].append(post.id)
    posts_dict["Score"].append(post.score)
    posts_dict["Total Comments"].append(post.num_comments)
    posts_dict["Post URL"].append(post.url)
    posts_dict["Label"].append('teaching')
    # Label Append
    

    

# Saving the data in a pandas dataframe
top_posts = pd.DataFrame(posts_dict)
top_posts

top_posts.to_csv("teaching.csv", index=True)
