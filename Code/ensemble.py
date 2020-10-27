import user_based
import item_based
import sys


def ensemble(userID=0, genre=None):

    if genre == 'N/A':
        userID = int(userID)
        genre = None
    else:
        userID = int(userID)
        genre = genre

    if userID > 610:
        print('Error: Wrong user ID!')
        sys.exit(1)

    rec_user_based = user_based.get_rec_user(userID, genre)
    rec_item_based = item_based.get_rec_item(userID, genre)

    return list(set(rec_user_based + rec_item_based))
