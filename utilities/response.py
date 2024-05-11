class Response():
    def __init__(self, statusCode, email, questionText, answerText, fileUrl, date, model) -> None:
        self.statusCode = statusCode
        self.email = email
        self.questionText = questionText
        self.answerText = answerText
        self.fileUrl = fileUrl
        self.date = date
        self.model = model

    def to_dict(self):
        return {
            'statusCode': self.statusCode,
            'email': self.email,
            'questionText': self.questionText,
            'answerText': self.answerText,
            'fileUrl': self.fileUrl,
            'date': self.date,
            'model': self.model
        }


class ErrorResponse():
    def __init__(self, statusCode, message) -> None:
        self.statusCode = statusCode
        self.message = message

    def to_dict(self):
        return {
            'statusCode': self.statusCode,
            'message': self.message
        }
