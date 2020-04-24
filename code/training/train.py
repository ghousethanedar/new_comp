"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import pickle
from azureml.core import Workspace
from azureml.core.run import Run
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn_pandas import DataFrameMapper
from sklearn.externals import joblib
import numpy as np
import json
from azure.storage.blob import BlockBlobService
from io import StringIO
import pandas as pd
import subprocess
from typing import Tuple, List


parser = argparse.ArgumentParser("train")
parser.add_argument(
    "--config_suffix", type=str, help="Datetime suffix for json config files"
)
parser.add_argument(
    "--json_config",
    type=str,
    help="Directory to write all the intermediate json configs",
)
args = parser.parse_args()

print("Argument 1: %s" % args.config_suffix)
print("Argument 2: %s" % args.json_config)

if not (args.json_config is None):
    os.makedirs(args.json_config, exist_ok=True)
    print("%s created" % args.json_config)

run = Run.get_context()
exp = run.experiment
ws = run.experiment.workspace


block_blob_service = BlockBlobService(account_name='mlworkshop',
                              account_key='kNYne8MDwB5flQnDnW6x8aX4MTNRZP0eraAEIM/040jQdC4gwUgd1ZR23MGzR0+8qMb1xcApb/n0WUzK1vXOwg==' )
# get data from blob storage in the form of bytes
blob_byte_data = block_blob_service.get_blob_to_bytes('final','temp1.csv')
# convert to bytes data into pandas df to fit scaler transform
s=str(blob_byte_data.content,'utf-8')
bytedata = StringIO(s)
df=pd.read_csv(bytedata)

x_df = df.drop(['Solo_Insulin'], axis=1)
y_df = df['Solo_Insulin']

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=0)

categorical = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id',
               'race', 'gender', 'age', 'change', 'readmitted']

numerical = ['patient_nbr', 'number_diagnoses', 'time_in_hospital', 'encounter_id',
             'num_lab_procedures', 'num_procedures', 'num_medications',
             'number_outpatient', 'number_emergency', 'number_inpatient',
             'Total_drugs', 'Solo_Insulin', 'diagnosis']

numeric_transformations = [([f], Pipeline(steps=[
    ('scaler', StandardScaler())])) for f in numerical]

categorical_transformations = [([f], OneHotEncoder(handle_unknown='ignore', sparse=False)) for f in categorical]

transformations = numeric_transformations + categorical_transformations

clf = Pipeline(steps=[('preprocessor', DataFrameMapper(transformations)),
                      ('classifier', RandomForestClassifier(max_depth=5))])
print("Running train.py")

clf.fit(X_train, y_train)

run.log("model", clf)

clf.fit(X_train, y_train)
preds = clf.predict(X_test)
run.log("accuracy", accuracy_score(y_test, preds))


# Save model as part of the run history
model_name = "solo_randomforest_model.pkl"
# model_name = "."

with open(model_name, "wb") as file:
    joblib.dump(value=clf, filename=model_name)

# upload the model file explicitly into artifacts
run.upload_file(name="./outputs/" + model_name, path_or_stream=model_name)
print("Uploaded the model {} to experiment {}".format(model_name, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)
print("Following files are uploaded ")
print(run.get_file_names())

# register the model
# run.log_model(file_name = model_name)
# print('Registered the model {} to run history {}'.format(model_name, run.history.name))

run_id = {}
run_id["run_id"] = run.id
run_id["experiment_name"] = run.experiment.name
filename = "run_id_{}.json".format(args.config_suffix)
output_path = os.path.join(args.json_config, filename)
with open(output_path, "w") as outfile:
    json.dump(run_id, outfile)

run.complete()