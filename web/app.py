import sqlite3
import os
import load_data.get_path as get_path
import json
import pandas as pd
from flask import Flask,render_template,request,redirect,session,url_for
import torch
from model.cold_start.cold_start_model import ColdStartSystem
from model.Wide.FeatureCross import FeatureCross

from web.sql.process_sql import build_new_sqlite,login_sql
from web.server.ColdStart import cold_start
from web.server.GetSeqs import reaction_data


# 指定模板目录为当前文件所在目录
app = Flask(__name__,template_folder="E:/Graduation_Project/MyRecommendSystem/web/font")
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
    session.pop("username",None)
    return redirect('/login')

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
    db_name = f"E:/Graduation_Project/MyRecommendSystem/web/sql/sqlite/{username}.sqlite"
    conn = sqlite3.connect(db_name)
    interaction = pd.read_sql("SELECT * FROM InterAction",conn)
    conn.close()
    if len(interaction[interaction["watch_ratio"]==1]) <=10:
        return "cold_start"
    else:
        return "rec_model"
    
@app.route('/cold_start')
def cold_start_rec():
    username = session['username']
    recommand_dict = cold_start(username)
    return json.dumps(recommand_dict)

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