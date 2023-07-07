from utils.Preprocess import Preprocess

sent = "내일 오전 10시에 탕수육 주문하고 싶어"

# Generate Preprocess object
p = Preprocess(userdic='../utils/user_dic.tsv')

# execute Preprocess
pos = p.pos(sent)

# print Keyword with tag
ret = p.get_keywords(pos, without_tag=False)
print(ret)

# print Keyword without tag
ret = p.get_keywords(pos, without_tag=True)
print(ret)