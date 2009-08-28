class potential_url_request:
  def __init__(self,text):
    self.text = text

  def is_url_request(self):
    #backward compatibility with Python 2.5
    try: from urlparse import parse_qs
    except: from cgi import parse_qs

    from urlparse import urlparse, urlunparse
    try:
      self.parsed = urlparse(self.text)
    except:
      return False
    self.file = self.parsed.path.split("?")[0]

    if "?" in self.parsed.path: #i.e., for file scheme, the query string is not
                          # supported. It shows up in the path instead.
      self.qs = parse_qs(self.parsed.path.split("?")[1])
    return True
