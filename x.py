import random
question = input(' question: should I be a doctor?')
random_number = random.randint(1,9)
if  random_number ==1:
  answer = "yes - definetly"
elif random_number ==2:
    answer = 'it is decidiely so'
elif random_number ==3:
  answer = 'without a doubt'
elif random_number ==4:
  answer = 'reply hazy, try again'
elif random_number ==5:
  answer = 'ask again later'
elif random_number ==6:
  answer = 'better not tell you now'
elif random_number ==7:
  answer = 'y sources say no'
elif random_number ==8:
  answer = 'outlook not so good'
elif random_number ==9:
  answer = 'very doubtful'
else:
  answer = 'ERROR'
print("Magic 8 ball:" + answer)