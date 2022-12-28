
import streamlit as st 
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
st.set_page_config(layout="wide")
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('입학전형 챗봇.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

st.header('부산소프트웨어마이스터고 챗봇')
st.subheader("안녕하세요 소마고 챗봇입니다.")

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('사용자 : ','')
    submitted = st.form_submit_button('전송')
    
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
    
if 'past' not in st.session_state:
    st.session_state['past'] = []
    
if submitted and user_input:
    embedding = model.encode(user_input)
    
    df['distance'] = df['embedding'].map(lambda x:cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]
    
    st.session_state.past.append(user_input)
    if answer['distance'] > 0.5:
        st.session_state.generated.append(answer['챗봇'])
    else :
        st.session_state.generated.append("무슨 말인지 모르겠어요.")
      
for i in range(len(st.session_state['past'])):
    st.markdown(
    """
    
    <div class="right-msg">
        <div
        class="msg-img">
        ></div>
        <div class="msg-bubble">
            <div class="msg-info">
            <div class="msg-info-time"></div>
            </div>
            <div class="right-bubble">
            <p>{0}</p></div>
        </div>
    </div>
    <div class="msg left-msg">
        <div
        class="msg-img"
        style="background-image: url(https://yt3.ggpht.com/5nJ5L79IMPPYFYVEaQ3)
        ></div>
        <div class="msg-bubble">
            <div class="msg-info">
            <div img="./부소마 프로필.jpeg">
            <div class="msg-info-name">소마고 챗봇</div>
            <div class="msg-info-time"></div>
            </div>
            <div class="left-bubble">
            <p>{1}</p>
        </div>
    """.format(st.session_state['past'][i], st.session_state['generated'][i]), unsafe_allow_html=True
    )  
#for i in range(len(st.session_state['past'])):
 #   message(st.session_state['past'][i],is_user=True, key= str(i) + '_user')
  #  if len(st.session_state['generated']) > i:
   #     message(st.session_state['generated'][i],key=str(i) + '_bot')
   
   


st.sidebar.title("BSSM")
st.sidebar.info(
    """
    [HomePage(https://school.busanedu.net/bssm-h/main.do)]
    [instagram(https://www.instagram.com/bssm.hs/)]
    """
)