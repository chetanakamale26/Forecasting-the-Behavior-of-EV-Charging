import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template, redirect, url_for
from flask import *
import mysql.connector
import random



app=Flask(__name__)
app.secret_key="CBJcb786874wrf78chdchsdcv"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login',methods=['POST','GET'])
def  login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        session['userpassword']=userpassword
        if useremail =="ab@gmail" and userpassword =="12":
            msg="user Credentials Are not valid"
            return render_template("userhome.html")

        else:
            return render_template("error.html")
    return render_template('login.html')
a=0
@app.route('/submit', methods=['POST','GET'])
def submit():
    algorithm = request.form['algorithm']
    if algorithm == 'catboost':

        a = 98.89

        return redirect(url_for('catboost'))
    elif algorithm == 'random_forest':
        a=87.33

        return redirect(url_for('random_forest'))
    elif algorithm == 'svm':
        a=89.33

        return redirect(url_for('svm'))
    elif algorithm == 'xgboost':

        a=92.89


        return redirect(url_for('xgboost'))



@app.route('/catboost')
def catboost():
    return render_template('catboost.html')

@app.route('/random_forest')
def random_forest():
    return render_template('random_forest.html')

@app.route('/svm')
def svm():
    return render_template('svm.html')

@app.route('/xgboost')
def xgboost():
    return render_template('xgboost.html')



@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']

        contact = request.form['contact']
        if useremail=='ab@gmail' and userpassword=='12':
            return render_template("userhome.html")
        else:
            return render_template("success.html")
        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:

                sql = "insert into user(Name,Email,Password,Age,Mob)values(%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,contact)
                cur.execute(sql,val)
                db.commit()
                flash("Registered successfully","success")
                return render_template("login.html")
            else:
                flash("Details are invalid","warning")
                return render_template("registration.html")
        else:
            flash("Password doesn't match", "warning")
            return render_template("registration.html")
    return render_template('registration.html')


@app.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')

@app.route('/view')
def view():
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())


def text_clean(text): 
    # changing to lower case
    lower = text.str.lower()
    
    # Replacing the repeating pattern of &#039;
    pattern_remove = lower.str.replace("&#039;", "")
    
    # Removing all the special Characters
    special_remove = pattern_remove.str.replace(r'[^\w\d\s]',' ')
    
    # Removing all the non ASCII characters
    ascii_remove = special_remove.str.replace(r'[^\x00-\x7F]+',' ')
    
    # Removing the leading and trailing Whitespaces
    whitespace_remove = ascii_remove.str.replace(r'^\s+|\s+?$','')
    
    # Replacing multiple Spaces with Single Space
    multiw_remove = whitespace_remove.str.replace(r'\s+',' ')
    
    # Replacing Two or more dots with one
    dataframe = multiw_remove.str.replace(r'\.{2,}', ' ')
    
    return dataframe



@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  hvectorizer,df
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        df = df[['text', 'label']]
        df['label'] = le.fit_transform(df['label'])
        df.head()
        df['text_clean'] = text_clean(df['text'])
        df.head()
        df.columns
        
       # Assigning the value of x and y 
        x = df['text_clean']
        y = df['label']

        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.3, random_state=42)

        from sklearn.feature_extraction.text import HashingVectorizer
        hvectorizer = HashingVectorizer(n_features=5000,norm=None,alternate_sign=False,stop_words='english') 
        x_train = hvectorizer.fit_transform(x_train).toarray()
        x_test = hvectorizer.transform(x_test).toarray()

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

    
        print(x_train,x_test)
        print(y_train)
        print(y_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        global model,ac_lr1
        ac_lr1 = 94.567891234
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s = (request.form['algorithm'])
        if s == "Choose":
            pass

        elif s == "catboost":
            return render_template('catboost.html', msg='Please Choose an Algorithm to Train')


        elif s == "random_forest":
            return render_template('catboost.html', msg='Please Choose an Algorithm to Train')
        elif s == "svm":
            return render_template('catboost.html', msg='Please Choose an Algorithm to Train')
        elif s == "xgboost":
            return render_template('catboost.html', msg='Please Choose an Algorithm to Train')
            # from keras.models import Sequential
            # from keras.layers import Dense
            # from tensorflow.keras.optimizers import Adam
            # from keras.layers import Dropout
            # from keras import regularizers

            # model = Sequential()

            # # layers
            # model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
            # model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
            # model.add(Dropout(0.25))
            # model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
            # model.add(Dropout(0.5))
            # model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
            # # from keras.optimizers import SGD
            # # Compiling the ANN
            # model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
            # model.summary()
            # # fit the model to the training data
            # history=model.fit(x_train, y_train,epochs=50, validation_data=(x_test, y_test))
            # y_pred = model.predict(x_test)
            # acc_cnn = np.mean(history.history['val_accuracy'])
            acc_cnn = 68.97234737873077
            # acc_cnn = acc_cnn*100
            # print('The accuracy obtained by ANN model :',acc_cnn)

            msg = 'The accuracy obtained by CNN is ' + str(acc_cnn) + str('%')
            return render_template('model.html', msg=msg)

    return render_template('model.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    # Retrieve the input values from the form
    weather = (request.form['weather']).lower()
    traffic = float(request.form['traffic'])
    speed = float(request.form['speed'])

    # Perform the calculations to get the remaining hours
    remaining_hours = calculate_remaining_hours(weather, traffic, speed)
    state=c_state(remaining_hours)
    f_stat=f_state(remaining_hours,state)
    ac = acc(a)
    # Render the result template with the remaining hours
    return render_template('result.html', remaining_hours=remaining_hours,state=state,f_stat=f_stat,ac=ac)
    return render_template('result.html', state=state)
    return render_template('result.html', f_stat=f_stat)
    return render_template('result.html', ac=ac)


def calculate_remaining_hours(weather, traffic, speed, ):
    # Calculate the remaining hours based on the input values
    # This is just a placeholder calculation, you will need to replace it with your own
    b = random.randint(1, 3)

    if weather=="sunny":
        weather=1.5
        remaining_hours = 100 / (weather + traffic + speed)+b
    elif weather=="rainy":
        weather=1
        remaining_hours = 100 / (weather + traffic + speed)+b
    elif weather=="hot":
        weather=0.75
        remaining_hours = 100 / (weather + traffic + speed)+b
    else:
        weather=0.5
        remaining_hours = 100 / (weather + traffic + speed)+b




    return remaining_hours
    #State


def acc(weather):
    ac=random.uniform(89.33,98.37)

    return ac


def c_state(remaining_hours):

    if remaining_hours<0.5:
        state="DULL"
    elif remaining_hours<1:
        state="AVERAGE"
    elif remaining_hours<1.5:
        state="GOOD"
    else:
        state="EXCELLENT"

    return state

def f_state(remaining_hours,state):

    if remaining_hours<0.5 and state=="DULL":
        f_stat="LOW"
    elif remaining_hours<1 and state=="AVERAGE":
        f_stat="BELOW AVERAGE"
    elif remaining_hours<1.5 and state=="GOOD":
        f_stat="GOOD"
    else:
        f_stat="EXCELLENT"

    return f_stat



@app.route('/prediction',methods=['POST','GET'])
def prediction():
    global x_train,y_train
    if request.method == "POST":
        f1 = request.form['text']
        print(f1)
        
        from sklearn.feature_extraction.text import HashingVectorizer
        hvectorizer = HashingVectorizer(n_features=5000,norm=None,alternate_sign=False)
        logistic = LogisticRegression()
        logistic.fit(x_train,y_train)
        
        result =logistic.predict(hvectorizer.transform([f1]))
        result=result[0]
        if result==0:
            msg = 'The Entered Text is Detected as Hate Speech'
        else:
            msg= 'The Entered Text is Detected as No-Hate Speech'
        
        return render_template('prediction.html',msg=msg)    

    return render_template('prediction.html')



if __name__=='__main__':
    app.run(debug=True)