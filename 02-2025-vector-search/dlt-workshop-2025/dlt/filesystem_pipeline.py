import dlt
import os


@dlt.resource(table_name="api_source_docs", max_table_nesting=0)
def get_data(locations):
    for location in locations:
        print("Browsing", location)
        files = os.listdir(location)
        for page in files:
            if not page.endswith(".txt"):
                continue
            print("Getting", page, "\n")
            with open(f"{location}/{page}", "r", encoding="utf-8") as f:  # Cambiato \ con /
                content = f.read()
            yield {
                "source_name": location.split("/")[-1], # Cambiato \ con /
                "node_set_category": location.split("/")[-1], # Cambiato \ con /
                "filename": page, # filename
                "content": content  # content of the page
            }


if __name__ == "__main__":
    # MY CUSTOM FILES
    source_locations = [
        "data_samples/api.slack.com",          # Cambiato \ con /
        "data_samples/developer.monday.com",   # Cambiato \ con /
        "data_samples/developer.paypal.com",   # Cambiato \ con /
        "data_samples/developer.ticketmaster.com"  # Cambiato \ con /
    ]

    pipeline = dlt.pipeline(
        pipeline_name="api_doc_sources_pipeline",
        dataset_name="api_sources",
        destination="filesystem"
    )

    info = pipeline.run(get_data(source_locations), loader_file_format="csv")
    print(info)