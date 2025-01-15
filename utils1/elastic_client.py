
from elasticsearch import Elasticsearch as ElasticsearchClient
from elasticsearch.helpers import bulk


class ElasticSearch:
    def __init__(self):
        # Elasticsearch konfigurieren
        config = {"http://localhost": "9200"}  # Elasticsearch läuft jetzt auf Port 9200

        self.es = ElasticsearchClient(
            [
                config,
            ],
            timeout=300,
        )
        self.last_scroll_id = None

    def create_index(self, name, mapping, replace=False):
        if replace:
            self.delete_index(name)
        print("Index wird erstellt, Name: ", name)
        self.es.indices.create(index=name, body=mapping)
        print("Index erfolgreich erstellt, Name: " + name)

    def delete_index(self, name):
        print("Index wird gelöscht, Name: ", name)
        self.es.indices.delete(index=name, ignore=[400, 404])
        print("Index erfolgreich gelöscht, Name: " + name)

    def index(self, documents, index_name, is_bulk=False):
        if is_bulk:
            try:
                response = bulk(
                    self.es, documents
                )
                print("\nAntwort:", response)
            except Exception as e:
                print("\nFehler:", e)

    def search(self, index, body):
        try:
            return self.es.search(index=index, body=body)
        except Exception as e:
            print("\nFehler:", e)

    def search_all_with_scroll(self, index, body):
        try:
            there_is_next_page = False
            resp = self.es.search(index=index, body=body, scroll="3m")
            self.last_scroll_id = resp["_scroll_id"]
            if len(resp["hits"]["hits"]) >= 10000:
                there_is_next_page = True
            while there_is_next_page:
                resp_scroll = self.es.scroll(
                    scroll="3m",
                    scroll_id=self.last_scroll_id,
                )
                self.last_scroll_id = resp_scroll["_scroll_id"]
                resp["hits"]["hits"].extend(resp_scroll["hits"]["hits"])
                if len(resp_scroll["hits"]["hits"]) >= 10000:
                    there_is_next_page = True
                else:
                    there_is_next_page = False

            if not there_is_next_page:
                self.last_scroll_id = None
                return resp
        except Exception as e:
            print("\nFehler:", e)

    def get_with_id(self, index, id_):
        try:
            return self.es.get(index=index, id=id_)
        except Exception as e:
            print("\nFehler:", e)

    def termvectors(self, index, body, id):
        try:
            return self.es.termvectors(index=index, body=body, id=id)
        except Exception as e:
            print("\nFehler:", e)
