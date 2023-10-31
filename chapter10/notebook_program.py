# Note 클래스 구현
class Note(object):
    def __init__(self, contents = None):
        self.contents = contents

    def write_contents(self, contents):
        self.contents = contents

    def remove_all(self):
        self.contents = ""

    def __str__(self):
        return self.contents
    
# Notebook 클래스 구현
class NoteBook(object):
    def __init__(self, title):
        self.title = title
        self.page_number = 1
        self.notes = {}

    def add_note(self, note, page = 0):
        if self.page_number < 300:
            if page == 0:
                self.notes[self.page_number] = note
                self.page_number += 1
            else:
                self.notes = {page : note}
                self.page_number += 1
        else:
            print("페이지가 모두 채워졌다.")
            
    def remove_note(self, page_number):
        if page_number in self.notes.keys():
            return self.notes.pop(page_number)
        else:
            print("해당 페이지는 존재하지 않는다.")

    def get_number_of_pages(self):
        return len(self.notes.keys())
    
from notebook import Note
from notebook import NoteBook

good_sentence = """세상 사는 데 도움이 되는 명언, 힘이 되는 명언, 용기를 주는 명언, 
위로되는 명언, 좋은 명언 모음 100가지. 자주 보면 좋을 것 같아 선별했습니다."""
note_1 = Note(good_sentence)

good_sentence = """삶이 있는 한 희망은 있다. - 키케로"""
note_2 = Note(good_sentence)

good_sentence = """하루에 3시간을 걸으면 7년 후에 지구를 한 바퀴 돌 수 있다. - 새뮤얼 존슨"""
note_3 = Note(good_sentence)

good_sentence = """행복의 문이 하나 닫히면 다른 문이 열린다. 그러나 우리는 종종 닫힌 문을
멍하니 바라보다가 우리를 향해 열린 문을 보지 못하게 된다. - 헬렌 켈러"""
note_4 = Note(good_sentence)

wise_saying_notebook = NoteBook("명언 노트")
wise_saying_notebook.add_note(note_1)
wise_saying_notebook.add_note(note_2)
wise_saying_notebook.add_note(note_3)
wise_saying_notebook.add_note(note_4)