from flask import Flask, request, render_template
from ensemble import ensemble
from user_based import get_rec_user
from item_based import get_rec_item
app = Flask(__name__, template_folder='.')


@app.route("/")
def landingpage():
    return render_template('landingPage.html')


@app.route('/recommend', methods=['POST'])
def recommend_movie():
    userid = request.form['userid']
    genre = request.form['genre']
    approach = request.form['approach']
    response = None
    if genre == 'N/A':
        genre = None
    if approach == 'item-based':
        response = get_rec_item(int(userid), genre)
    elif approach == 'user-based':
        response = get_rec_user(int(userid), genre)
    elif approach == 'ensemble':
        response = ensemble(userid, genre)
    return render_template('response.html', data=response)


if __name__ == "__main__":
    app.run()
