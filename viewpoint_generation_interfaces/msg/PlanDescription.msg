# A plan file as its producer describes it (see DescribePlans).
string path
bool valid                # false: unreadable, or not a plan
string stage
int32 stage_rank          # higher = further through the pipeline
string summary_json
string summary_text
string change_token       # '' when the file does not say
string generated_at       # ISO-8601 UTC
string message
