import face_recognition
from os import listdir
import lip_tracker
import pickle

def main():

    # We're findinging the lipstick shades for one person at a time, that means we have to supply an example of that person
    known_image = face_recognition.load_image_file("images/riri.jpg")

    # Next, we have designate the folder with all of our scraped instagram stuff
    folder_name='/mnt/CABC33CABC33B035/insta/riri'

    files=listdir(folder_name)

    # create the storage for all of our makeup shades
    storage=[]
    with open('riri.pkl', 'wb') as f:
        pickle.dump(storage, f)

    for file in files:
        if file.split('.')[1]=='jpg': #only images plz

            # compare known and unknown images, see if our target person is present
            unknown_image = face_recognition.load_image_file(folder_name+"/"+file)
            kim_encoding = face_recognition.face_encodings(known_image)[0]
            unknown_encoding = face_recognition.face_encodings(unknown_image)

            for z in range(len(unknown_encoding)):
                results = face_recognition.compare_faces([kim_encoding], unknown_encoding[z])
                if results[0]==1:
                    collist=lip_tracker.extraction(unknown_image,z)
                    print(str(file)+': '+str(collist[0]))
                    storage.append(collist[0])
            with open('riri.pkl', 'wb') as f:    # We cache after every loop : Always Be Caching
                 pickle.dump(storage, f)
if __name__ == '__main__':
    main()
