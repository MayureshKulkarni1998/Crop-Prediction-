from flask import Flask, render_template, request, redirect, url_for, session
from datetime import date

from fpdf import FPDF
from sklearn import metrics

app = Flask(__name__)
import mysql.connector
app.secret_key = "abc"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap


irisData = load_iris()

conn=mysql.connector.connect(host="localhost",user="root",password="",database="demo")
cursor=conn.cursor()

@app.route("/")
def hello_world():

    return render_template('index.html')

@app.route("/login")
def login():

    return render_template('login.html')

@app.route("/adminreg")
def adminreg():
    cursor.execute("""SELECT * FROM `register`""")
    data1 = cursor.fetchall()
    return render_template('adminreg.html',data1=data1)

@app.route("/admincrop")
def admincrop():
    cursor.execute("""SELECT * FROM `croppredict`""")
    data1 = cursor.fetchall()
    return render_template('admincrop.html',data1=data1)


@app.route("/admincontact")
def admincontact():
    cursor.execute("""SELECT * FROM `contact`""")
    data1 = cursor.fetchall()
    return render_template('admincontact.html',data1=data1)

@app.route("/admindash")
def admindash():
    cursor.execute("""SELECT COUNT(*) FROM `register`""")
    data1 = cursor.fetchall()
    for row in data1:
        trecord = row[0]

    cursor.execute("""SELECT COUNT(*) FROM `croppredict`""")
    data = cursor.fetchall()
    for row in data:
        tcrop = row[0]

    return render_template('admindash.html',trecord=trecord,tcrop=tcrop)


@app.route("/register")
def register():

        return render_template('register.html')


@app.route("/crop")
def crop():
    if 'username' in session:
        num = session['username']

        cursor.execute("""SELECT * FROM register WHERE phone = '{}'""".format(num))
        data1 = cursor.fetchall()
        for row in data1:
            fname = row[1]

        temp = session['temp']
        soil = session['soil']
        cursor.execute("""SELECT * FROM test WHERE temp = '{}' and soil = '{}' """.format(temp,soil))
        data = cursor.fetchall()

        return render_template('crop.html',data=data,num=num,fname=fname)

    else:

        return redirect('/login')


@app.route("/prevcrop")
def prevcrop():
        if 'username' in session:
            num = session['username']

            cursor.execute("""SELECT * FROM register WHERE phone = '{}'""".format(num))
            data1 = cursor.fetchall()
            for row in data1:
                fname = row[1]


            cursor.execute("""SELECT * FROM croppredict WHERE phone = '{}'""".format(num))
            data = cursor.fetchall()

            return render_template('prevcrop.html',data=data,num=num,fname=fname)
        else:
            return redirect('/login')





@app.route("/dashboard")
def dashboard():
        if 'username' in session:
            num = session['username']

            cursor.execute("""SELECT * FROM register WHERE phone = '{}'""".format(num))
            data = cursor.fetchall()
            for row in data:
                fname = row[1]
                dob = row[2]
                phone = row[3]
                email = row[4]
                fsize = row[10]
                stype = row[11]
                flocation = row[12]
            return render_template('dashboard.html', number=num, fname=fname, dob=dob, phone=phone, email=email,
                                   fsize=fsize, stype=stype, flocation=flocation)

        else :
            return redirect('/login')

@app.route("/updateprofile")
def updateprofile():
        if 'username' in session:
            num = session['username']

            cursor.execute("""SELECT * FROM register WHERE phone = '{}'""".format(num))
            data = cursor.fetchall()
            for row in data:
                fname = row[1]
                dob = row[2]
                phone = row[3]
                email = row[4]
                country = row[5]
                state = row[6]
                district = row[7]
                address = row[8]
                pincode = row[9]
                fsize = row[10]
                stype = row[11]
                flocation = row[12]
            return render_template('updateprofile.html',number = num,fname = fname, dob=dob, phone=phone, email=email,
                                   fsize=fsize, stype=stype, flocation=flocation,country=country,state=state,
                               district=district,address=address,pincode=pincode)


        else:

            return redirect('/login')



@app.route("/croppredict")
def croppredict():
        if 'username' in session:
                num = session['username']

                cursor.execute("""SELECT * FROM register WHERE phone = '{}'""".format(num))
                data = cursor.fetchall()
                for row in data:
                    fname = row[1]

                return render_template('croppredict.html',number = num,fname = fname)

        else:

                return redirect('/login')

@app.route('/add_user', methods=['POST'])
def add_user():
    fname = request.form.get('name')
    lname= request.form.get('dob')
    email= request.form.get('email')
    phone = request.form.get('phone')
    country = request.form.get('country')
    state= request.form.get('state')
    district = request.form.get('district')
    address = request.form.get('address')
    pincode = request.form.get('pincode')
    fsize = request.form.get('fsize')
    stype = request.form.get('stype')
    flocation = request.form.get('flocation')
    pass1 = request.form.get('pass1')
    pass1 = request.form.get('pass2')



    cursor.execute("""INSERT INTO `register`(`id`, `fname`, `lname`, `phone`,`email`,`country`, `state`, `district`, `address`, `pincode`, `fsize`, `stype`, `flocation`,`pass1`,`pass2`) VALUES ('','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}')"""
                   .format(fname,lname,phone,email,country,state,district,address,pincode,fsize,stype,flocation,pass1,pass1))
    conn.commit()
    #users=cursor.fetchall()

    #if len(users)>0:
        #return render_template('home.html')
    #else:
        #return render_template('login.html')

    #return render_template(url_for("login"))

    return redirect('/login')



@app.route('/login_validation', methods=['POST'])
def login_validation():
    num= request.form.get('username')
    pas= request.form.get('pass')

    if num == "admin" and pas == "admin":

        return redirect('/admindash')

    else:


        cursor.execute("""SELECT * FROM register WHERE phone = '{}' AND pass1 = '{}' """.format(num,pas))

        users=cursor.fetchall()

    if len(users)>0:

        session['username'] = request.form.get('username')

        if 'username' in session:
            num = session['username']

        cursor.execute("""SELECT * FROM register WHERE phone = '{}'""".format(num))
        data = cursor.fetchall()
        for row in data:
            fname = row[1]
            dob = row[2]
            phone = row[3]
            email = row[4]
            fsize = row[10]
            stype = row[11]
            flocation = row[12]

        return redirect('/dashboard')
        #return render_template('dashboard.html',number = num,fname = fname,dob=dob,phone=phone,email=email,fsize=fsize,stype=stype,flocation=flocation)
    else:
        # print("Username/Password invalid. Please enter proper credentials")
        return redirect('/login')


@app.route('/logout')
def logout():
    session.pop('username',None)
    session.pop('temp', None)
    session.pop('soil', None)
    return redirect('/login')

@app.route('/pdf/<temp>/<soil>/<crop>', methods=['POST','get'])
def pdf(temp,soil,crop):
    num = session['username']
    cursor.execute("""SELECT * FROM register WHERE phone = '{}'""".format(num))
    data = cursor.fetchall()
    for row in data:
        fname = row[1]
        phone = row[3]
        email = row[4]
    pdf = FPDF()
    pdf.add_page()

    page_width = pdf.w - 2 * pdf.l_margin

    pdf.set_font('Times', 'B', 14.0)
    pdf.cell(page_width, 0.0, 'Crop Predication System', align='C')

    pdf.ln(10)

    pdf.set_font('Courier', 'B', 12)

    col_width = page_width / 4

    pdf.ln(1)

    th = pdf.font_size
    # dt = datetime.now()
    today = date.today()

    # date = dt.date()
    pdf.cell(190, 10, ('Date: %s' % today), ln=1, align="R")
    pdf.cell(190, 10, txt="Crop Details:", ln=1, align="C")
    # pdf.cell(200, 10, txt="pregnant_times:", ln=1, align="C")
    pdf.cell(190, 10, ('Full name: %s' % fname), ln=1, align="L", border=1)
    pdf.cell(190, 10, ('Phone Number: %s' % phone), ln=1, align="L", border=1)
    pdf.cell(190, 10, ('Email: %s' % email), ln=1, align="L", border=1)
    pdf.cell(190, 10, ('Soil: %s' % temp), ln=1, align="L", border=1)
    pdf.cell(190, 10, ('Temp: %s' % soil), ln=1, align="L", border=1)
    pdf.cell(190, 10, ('Crop: %s' % crop), ln=1, align="L", border=1)
    pdf.cell(190, 10, txt="Thank You!!", ln=1, align="C")
    # pdf.output("Report.pdf")
    session['files'] = pdf.output("Report.pdf")

    pdf.ln(10)

    pdf.set_font('Times', '', 10.0)
    pdf.cell(page_width, 0.0, '- end of report -', align='C')

    # And you return a text or a template, but if you don't return anything
    # this code will never work.

    return redirect('/crop')


@app.route('/updateinfo', methods=['POST'])
def updateinfo():
    num = session['username']

    fname = request.form.get('name')
    lname = request.form.get('dob')
    email = request.form.get('email')
    country = request.form.get('country')
    state = request.form.get('state')
    district = request.form.get('district')
    address = request.form.get('address')
    pincode = request.form.get('pincode')
    fsize = request.form.get('fsize')
    stype = request.form.get('stype')
    flocation = request.form.get('flocation')

    cursor.execute("""UPDATE `register` SET `fname`='{}',`lname`='{}',`email`='{}',`country`='{}',`state`='{}',`district`='{}',`address`='{}',`pincode`='{}',`fsize`='{}',`stype`='{}',`flocation`='{}' WHERE `phone`='{}'"""
                   .format(fname,lname,email,country,state,district,address,pincode,fsize,stype,flocation,num))
    conn.commit()


    return redirect('/updateprofile')



@app.route('/predict', methods=['POST'])
def predict():
    if 'username' in session:
            num = session['username']


            session['temp'] = request.form.get('temp')
            session['soil'] = request.form.get('soil')
            #cursor.execute("""INSERT INTO `croppredict`(`id`, `phone`, `temp`, `soil`) VALUES ('','{}','{}','{}')""".format(num,temp,soil))
            #conn.commit()
            return redirect('/crop')

    else:

        return redirect('/login')


@app.route('/savecrop/<temp>/<soil>/<crop>', methods=['POST','get'])
def savecrop(temp,soil,crop):
    if 'username' in session:
            num = session['username']
            today = date.today()
            cursor.execute("""INSERT INTO `croppredict`(`id`, `phone`, `temp`, `soil`,`crop`,`date`) VALUES ('','{}','{}','{}','{}','{}')""".format(num,temp,soil,crop,today))
            conn.commit()
            return redirect('/crop')

    else:

        return redirect('/login')

@app.route('/contact', methods=['POST'])
def contact():
    name = request.form.get('name')
    email = request.form.get('email')
    subject = request.form.get('subject')
    message = request.form.get('message')

    cursor.execute("""INSERT INTO `contact`(`id`,`name`,`email`,`subject`,`message`) VALUES ('','{}','{}','{}','{}')""".format(name,email,subject,message))
    conn.commit()

    return redirect('/')




if __name__ == '__main__':
    app.run(debug=True)

''' 
# @app.route('predictCrop')
# def predictCrop():
#     # Create feature and target arrays
#     X = request.form.get('temp')
#     y = request.form.get('soil')
#
#     # Split into training and test set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0)
#
#     neighbors = np.arange(1, 9)
#     train_accuracy = np.empty(len(neighbors))
#     test_accuracy = np.empty(len(neighbors))
#
#     # Loop over K values
#     for i, k in enumerate(neighbors):
#         knn = KNeighborsClassifier(n_neighbors=k)
#         knn.fit(X_train, y_train)
#
#         # Compute traning and test data accuracy
#         train_accuracy[i] = knn.score(X_train, y_train)
#         test_accuracy[i] = knn.score(X_test, y_test)
#         data1= test_accuracy[i]
#     # Generate plot
#     plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
#     plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')
#
#     plt.legend()
#
#     plt.show(data1)
'''
#importing dataset
dataset = pd.read_csv('Book1.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training KNN model
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Making confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix: \n",cm)
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
# print("\nAccuracy: ",as)

# # Predicting a new result
# print(f"Prediction on {temp},{soil}:\n",classifier.predict(sc.transform([[temp,soil]])))

'''#Visualizing Training set data
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('K-NN (Training set)')
plt.xlabel('Temperature')
plt.ylabel('Moisture')
plt.legend()
plt.show()

#Visualizing Testing set data
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('K-NN (Test set)')
plt.xlabel('Temperature')
plt.ylabel('Moisture')
plt.legend()
plt.show()
'''
if __name__ == '__main1__':
    app.run(debug=True)





