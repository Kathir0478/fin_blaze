import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import google.generativeai as genai 
import yt_dlp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app= Flask(__name__)

load_dotenv()
api_key = os.getenv("GEN_AI_KEY")
genai.configure(api_key=api_key)
model=genai.GenerativeModel("gemini-1.5-pro-latest")
ydl_opts = {
    "quiet": True,
    "extract_flat": True, 
    "skip_download": True,
    "default_search": "ytsearch250"
}

@app.route('/process',methods=['POST'])
def process_prompt():
    data=request.json
    user_prompt=data.get("prompt","")

    response = model.generate_content(f"Generate a structured response in the format: 'Search Query: [generalized search query]' followed by 'User Needs: [distinct correlated single length tokens to the search query(10)]'. User prompt: {user_prompt} with proper formatting")
    
    text_response=response.text
    search_query= text_response.split("Search Query:")[1].split("User Needs:")[0].strip()
    user_needs=text_response.split("User Needs")[1].strip().split(", ")
    user_needs=" ".join(user_needs)
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        search_results = ydl.extract_info(f"ytsearch250:{search_query}", download=False)

    metadata=[]
    for video in search_results.get("entries",[]):
        title= video.get("title","").lower()
        view_count= video.get("view_count", 0)
        url=video.get("url","")
        vectorizer=CountVectorizer().fit_transform([user_needs,title])
        similarity_matrix=cosine_similarity(vectorizer)
        title_similarity_score=similarity_matrix[0][1]
        if "shorts/" in url:
            continue
        normalized_view_count = np.log1p(view_count) / 10 
        final_score = (0.8 * title_similarity_score) + (0.2 * normalized_view_count)
        metadata.append({
            "title": video.get("title", "Unknown Title"),
            "url": url,
            "view_count": view_count,
            "final_score":final_score 
        })
    filtered_data=[]
    metadata.sort(key=lambda x: x["final_score"], reverse=True)
    for video in metadata[:20]:
        filtered_data.append(video)
    return jsonify({"search_query":search_query,"user_needs":user_needs,"filtered_videos":filtered_data})

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)