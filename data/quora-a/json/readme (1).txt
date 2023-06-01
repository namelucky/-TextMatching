field	explanation
guid	unique id
qu_id/qu	the id and text of the user question
qa_id/qa	the id and text of the archived question
ans_id/ans	the id and text of the archived answer
is_duplicate	whether `qa' is similar with `qu'
is_correspond	whether `ans' is the answer of `qa'
is_useful	is True only when `is_duplicate' is True and `is_correspond' is True
