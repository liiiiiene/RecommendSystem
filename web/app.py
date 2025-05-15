import sqlite3
import os
import load_data.get_path as get_path
import json
import pandas as pd
from flask import Flask,render_template,request,redirect,session,url_for
from model.Wide.FeatureCross import FeatureCross
from web.sql.process_sql import build_new_sqlite,login_sql
from web.server.ColdStart import cold_start
from web.server.GetSeqs import reaction_data,get_seq,process_seq
from model.concat.WideAndDeep import WideAndDeep
from model.System.RecommendSystem import RecommendSystem
from web.server.RecSystem import get_rec_dict,explain_llm

# 指定模板目录为当前文件所在目录
app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)),"font"),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)),"font"),
            static_url_path='/css'
            )
app.secret_key = "123"  # 设置session密钥，实际应用中应使用复杂的随机字符串


@app.route("/")
def root():
    return render_template("root.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route('/main')
def main():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template("main.html", username=session['username'])

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/logout')
def logout():
    if len(session)!=0 and session["username"] is not None:
        session.pop("username",None)
    if len(session)!=0 and (session['llm_dic'] is not None or session['explain_dic'] is not None):
        session.pop('llm_dic',None)
        session.pop('explain_dic',None)
    return redirect('/login')

@app.route('/return_root')
def return_root():
    if len(session)!=0 and session["username"] is not None:
        session.pop("username",None)
    if len(session)!=0 and (session['llm_dic'] is not None or session['explain_dic'] is not None):
        session.pop('llm_dic',None)
        session.pop('explain_dic',None)
    return redirect('/')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/submitRegister',methods=['POST'])
def RegisterSubmit():
    user_name = request.form.get("username")
    password = request.form.get("password")
    ret = build_new_sqlite(user_name,password)
    if ret:
        # 注册成功后，设置session
        session['username'] = user_name
        return "true"
    else:
        return "false"

@app.route('/submitLogin',methods=['POST'])
def LoginSubmit():
    user_name = request.form.get("username")
    password = request.form.get("password")
    ret = login_sql(user_name,password)
    if ret:
        session['username'] = user_name
        return "true"
    else:
        return "false"

@app.route('/select_model')
def select_model():
    username = session["username"]
    db_name = os.path.join(get_path.sqlite_folder_path,f"{username}.sqlite")
    conn = sqlite3.connect(db_name)
    interaction = pd.read_sql("SELECT * FROM InterAction",conn)
    user_id = pd.read_sql("""SELECT index_id FROM UserInfo""", conn).loc[0].values
    if len(interaction[interaction["watch_ratio"]==1]) >10:
        u_s = get_seq(username)
    else:
        u_s = None
    conn.close()
    if u_s is None or len(u_s[user_id]) <= 10:
        return "cold_start"
    else:
        process_seq(username)
        llm_dic = get_rec_dict(username)
        explain_dic = explain_llm(username)
        session['llm_dic'] = json.dumps(llm_dic)
        session['explain_dic'] = json.dumps(explain_dic)
        return "rec_model"
    
@app.route('/cold_start')
def cold_start_rec():
    username = session['username']
    recommand_dict = cold_start(username)
    return json.dumps(recommand_dict)

@app.route('/rec_model')
def rec_model():
    return session['llm_dic']

@app.route('/explain_llm')
def get_explain():
    return session['explain_dic']

@app.route('/save_interaction',methods=["POST"])
def save_interaction():
    username = session["username"]
    try:
        data = request.get_json()
        video_id = eval(data['video_id']) if not isinstance(data['video_id'],str) else data['video_id']
        timestamp = data['timestamp']
        watch_ratio = data['watch_ratio']
    except:
        return json.dumps({"status":"error","meg":"数据不完整"})
    
    try:
        reaction_data(username,video_id,timestamp,watch_ratio)
        return json.dumps({"status":"success"})
    except:
        return json.dumps({"status":"error","meg":"存入数据库错误"})


if __name__ == "__main__":
    app.run(debug=True)
    # other_recommandation("admin")