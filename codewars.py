import re
import numpy as np
import itertools

str_in = 'Stop gninnipS My sdroW!'


def spin_str(str_in):
    res = ''
    for match in re.finditer('\w+|\W+', str_in):
        sym = match.group(0)
        res += sym[::-1] if len(sym) >= 5 else sym

    return res


def decode_morse(morse_code):
    # Remember - you can use the preloaded MORSE_CODE dictionary:
    # For example:
    # MORSE_CODE['.-'] = 'A'
    # MORSE_CODE['--...'] = '7'
    # MORSE_CODE['...-..-'] = '$'

    MORSE_CODE_DICT = {'A': '.-', 'B': '-...',
                       'C': '-.-.', 'D': '-..', 'E': '.',
                       'F': '..-.', 'G': '--.', 'H': '....',
                       'I': '..', 'J': '.---', 'K': '-.-',
                       'L': '.-..', 'M': '--', 'N': '-.',
                       'O': '---', 'P': '.--.', 'Q': '--.-',
                       'R': '.-.', 'S': '...', 'T': '-',
                       'U': '..-', 'V': '...-', 'W': '.--',
                       'X': '-..-', 'Y': '-.--', 'Z': '--..',
                       '1': '.----', '2': '..---', '3': '...--',
                       '4': '....-', '5': '.....', '6': '-....',
                       '7': '--...', '8': '---..', '9': '----.',
                       '0': '-----', ',': '--..--', '.': '.-.-.-',
                       '?': '..--..', '/': '-..-.', '-': '-....-',
                       '(': '-.--.', ')': '-.--.-', '$': '...-..-',
                       '&': '.-...', ':': '---...',
                       'SOS': '...---...', '!': '-.-.--', '+': '.-.-.',
                       '"': '.-..-.', '_': '..--.-', '\'': '.----.',
                       ';': '-.-.-.', '=': '-...-', '@': '.--.-.', '¿': '..-.-',
                       '¡': '--...-'
                       }

    Reverse_MORSE_CODE_DICT = {value: key for key, value in MORSE_CODE_DICT.items()}

    morse_code = morse_code.strip()
    words_list = [word.split(" ") for word in morse_code.split("   ")]

    words_list = [[Reverse_MORSE_CODE_DICT[sym] for sym in word] for word in words_list]
    res = ''
    for word in words_list:
        res += "".join(word) + " "

    # return ' '.join(''.join(MORSE_CODE[letter] for letter in word.split(' ')) for word in morseCode.strip().split('   '))
    return res.strip()


def swap_el(pointer):
    if pointer.next != None:
        if pointer.data > pointer.next.data:
            temp = pointer.next.data
        pointer.next.data = pointer.data
        pointer.data = temp
        if pointer.next.next != None:
            swap_el(pointer.next)


class WordDictionary:
    def __init__(self):
        self.dict_words = {}

    def add_word(self, word):
        self.dict_words[word.lower()] = word

    def __deep_search(self, word):
        template = re.compile(word)
        for word in self.dict_words.values():
            if re.fullmatch(template, word):
                return True
        return False

    def search(self, word):
        return True if self.dict_words.get(word.lower()) else self.__deep_search(word.lower())


def user_contacts(data):
    dict = {}
    for item in data:
        dict.setdefault(item[0], item[1]) if len(item) > 1 else dict.setdefault(item[0], None)

    return dict


def user_contacts_best(data):
    return {contact[0]: contact[1] if len(contact) > 1 else None
            for contact in data}


from itertools import zip_longest


def user_contacts_zip(data):
    print(*data)
    print(data)
    return dict(zip(*zip_longest(*data)))


def transpose_list(list_of_lists):
    print('list', list_of_lists)
    print('*list', *list_of_lists)
    print('zip*list', list(zip(*list_of_lists)))
    return [
        list(row)
        for row in zip(*list_of_lists)
    ]


# >>> date_info = {'year': "2020", 'month': "01", 'day': "01"}
# >>> track_info = {'artist': "Beethoven", 'title': 'Symphony No 5'}
# >>> filename = "{year}-{month}-{day}-{artist}-{title}.txt".format(
# ...     **date_info,
# ...     **track_info,
# ... )
# >>> filename
# '2020-01-01-Beethoven-Symphony No 5.txt'

def get_multiple(*keys, dictionary, default=None):
    return [
        dictionary.get(key, default)
        for key in keys
    ]


def test(dict):
    print(**dict)


# create_phone_number([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]) # => returns "(123) 456-7890"
def create_phone_number(n):
    return '({0}{1}{2}) {3}{4}{5}-{6}{7}{8}{9}'.format(*n)


# There are two lists, possibly of different lengths.
# The first one consists of keys, the second one consists of values.
# Write a function createDict(keys, values) that returns a dictionary created from keys and values.
# If there are not enough values, the rest of keys should have a None (JS null)value.
# If there not enough keys, just ignore the rest of values.

def createDict(keys, values):
    result_dict = dict(zip_longest(keys, values))
    if result_dict.get(None, False):
        del result_dict[None]
    return result_dict

def createDict_alt(keys, values):
    return dict(zip(keys, values + [None]*(len(keys) - len(values))))
#########################

# Python dictionaries are inherently unsorted.
# So what do you do if you need to sort the contents of a dictionary?
# Create a function that returns a sorted list of (key, value) tuples
# (Javascript: arrays of 2 items).
# The list must be sorted by the value and be sorted largest to smallest.
def sort_dict(d):
    return  sorted(d.items(),key=lambda x:x[1],reverse=True)

# In this kata, you will take the keys and values of a dict and swap them around.
# You will be given a dictionary, and then you will want to return a dictionary with the old values as the keys, and list the old keys as values under their original keys.
# For example, given the dictionary: {'Ice': 'Cream', 'Age': '21', 'Light': 'Cream', 'Double': 'Cream'},
# you should return: {'Cream': ['Ice', 'Double', 'Light'], '21': ['Age']}

def switch_dict(dic):
    res={}
    [res.setdefault(value,[]).append(key)  for key, value in dic.items()]
    return res


def chain(init_val, functions):
    print(init_val, functions)
    if len(functions) == 0:
        print('init_val=', init_val)
        return init_val
    else:
        return chain(functions[0](init_val), functions[1:])

def add10(x): return x + 10
def mul30(x): return x * 30

# Write a function that when given a URL as a string, parses out just the domain name and returns it as a string.
# For example:
# * url = "http://github.com/carbonfive/raygun" -> domain name = "github"
# * url = "http://www.zombie-bites.com"         -> domain name = "zombie-bites"
# * url = "https://www.cnet.com"                -> domain name = cnet"






def domain_name(url):

    #match=re.search('\w+://(\S+?)\.|www\.(\S+?)\.',url)
    match = re.search('(?:http://|https://){0,1}(?:www\.{0,1}){0,1}(\S+?)\.{1}', url )

    return (match[1]) if match[1] else ( match[2] if match[2] else None)

def solution(roman):
    roman_dict={'I':1,
                'V':5,
                'X':10,
                'L':50,
                'C':100,
                'D':500,
                'M':1000
                }
    roman_list= np.array([roman_dict[symb] for symb in roman])
    roman_list_shift=np.append(roman_list[1:],0)
    is_bigger=roman_list>=roman_list_shift
    is_smaller=roman_list<roman_list_shift
    return roman_list[is_bigger].sum()-roman_list[is_smaller].sum()

# https://www.codewars.com/kata/54521e9ec8e60bc4de000d6c/train/python
def max_sequence_non_opt(arr):
    arr=np.array(arr)
    max_sub_list=np.array([])
    max_sum=0
    for f in range(len(arr)+1):
        for i in range(f,len(arr)+1):
            sub_list=arr[f:i]

            cur_sum=np.sum(sub_list)
            if cur_sum>max_sum:
                max_sum=cur_sum
                max_sub_list=sub_list


    return max_sub_list, max_sum

def max_sequence_opt(arr):
    # алгоритм Алгоритм Кадане (динамическое программирование)
    arr=np.array(arr)
    zero=np.array([0]*len(arr))
    flag=np.sum(arr>zero)
    print(arr)
    if len(arr) > 1 and flag:
        local_sum = 0
        global_sum = 0
        for i in arr:
            local_sum = max(i, local_sum + i)
            global_sum = max(global_sum, local_sum)
    else:
        global_sum = 0
    return global_sum

def max_sequence_opt_refact(arr):
    #https: // math4everyone.info / blog / 2020 / 12 / 29 / poisk - maksimalnoj - summy - posledovatelnyh - elementov - massiva - algoritm - kadane - dinamicheskoe - programmirovanie /

    local_sum = 0
    global_sum = 0
    for i in arr:
        local_sum = max(i, local_sum + i)
        global_sum = max(global_sum, local_sum)

    return global_sum

def max_sequence(arr):
    print(arr)
    arr = np.array(arr)
    max_sub_list = np.array([])
    if  len(arr):
        max_sum = np.array(arr).sum()
    else:
        return 0
    print('max_sum', max_sum)

    min_index=np.argmin(arr)
    print('min index=',min_index, 'min val', arr[min_index])
    if np.array(arr[:min_index]).sum() > max_sum or np.array(arr[min_index+1:]).sum() > max_sum:
        print('left max',arr[:min_index].sum())
        print('right max',arr[min_index+1:].sum() )
        left_sum=arr[:min_index].sum()
        right_sum=arr[min_index+1:].sum()

        if left_sum > max_sum and left_sum < arr[min_index] and left_sum>right_sum:
            return max_sequence(arr[:min_index])
        else:
            if right_sum > max_sum and right_sum.sum()< arr[min_index] and left_sum<right_sum:
                return max_sequence(arr[min_index+1:])
            else:
                return max_sequence_non_opt(arr)


    else:
        return max_sequence_non_opt(arr)



#https://www.codewars.com/kata/5526fc09a1bbd946250002dc/train/python

def my_print(func):
    def wrappedFunc(arg):
        res=func(arg)
        print('res=',res)
        return res
    return wrappedFunc

@my_print
def find_outlier(integers):
    integers = np.array(integers)
    odds = np.array([2] * len(integers))
    check=integers%odds
    return int(integers[check==1]) if check.sum() == 1 else  int(integers[check==0])



@my_print
def foo():
    pass

# прохід матриці по равлику поворот матриці

def snail(arr):

    if type(arr)!=type(np.array([])):
        arr=np.array(arr)

    def __snail_out(arr):
        res=[]
        if len(arr[0])>1:

            res=res+list(arr[0,0:-1])+list(arr[:,-1])
            arr=arr[1:, 0:-1]
            arr=np.array(list(zip(*arr))[::-1])
            arr = np.array(list(zip(*arr))[::-1])
            return res+__snail_out(arr)

        else:

            res.append(arr[0,0])
            return res

    if np.shape(arr) == np.shape([[]]):
        return []
    else:
        return __snail_out(arr)

def snail(array):
    m = []
    array = np.array(array)
    while len(array) > 0:
        m += array[0].tolist()
        array = np.rot90(array[1:])
    return m

if __name__ == '__main__':
    # data = [["Grae Drake", 98110], ["Bethany Kok"], ["Alex Nussbacher", 94101], ["Darrell Silver", 11201]]
    # print(user_contacts_zip(data))
    # print(transpose_list([[1, 4, 7], [2, 5, 8], [3, 6, 9]]))
    #
    # fruits = {'lemon': [1,2,3], 'orange': 'orange', 'tomato': 'red'}
    # fruits_1 = {'lemon': [1, 4], 'orange': [2], 'tomato': 'red'}
    # #get_multiple('lemon', 'tomato', 'squash', dictionary=fruits, default='unknown')
    # #test(fruits)
    #
    # print('dict=', data)
    # print('*dict=', *data)
    #
    # print('dict=', list(zip_longest(*data)))
    # print('dict=', dict(zip(*zip_longest(*data))))
    # num=[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]  # => returns "(123) 456-7890"
    # print(create_phone_number(num))
    # dict_in={'Ice': 'Cream', 'Age': '21', 'Light': 'Cream', 'Double': 'Cream'}
    # print(switch_dict(dict_in))
    # print(dict_in)

    #print(chain(50, [mul30]))

    # url = "www.xakep.ru"
    # print(domain_name(url))
    # url = "https://google.com"
    # print(domain_name(url))
    #
    # url = "http: // google.co.jp"
    # print(domain_name(url))
    # print(solution('XXI'))
    #arr=[160, 3, 1719, 19, 11, 13, -21]
    #find_outlier(arr)
    arr = [[1, 2, 3, 4],
           [5, 6, 7, 8],
           [9, 10, 11, 12],
           [13, 14, 15, 16]]
    print(snail([[]]))




