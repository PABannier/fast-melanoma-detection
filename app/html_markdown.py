app_off = """
<div class="alert alert-danger" role="alert">
   <b> STATUS: </b> IMAGE NOT UPLOADED
</div>
"""


app_off2 = """
<div class="alert alert-block alert-warning">
  <b> Please Upload or Select an Image </b>
</div>
"""


model_predicting = """
<div class="alert alert-info" role="alert">
  Model predicting...
</div>
"""


loading_bar = """
<div class="progress">
  <div
    class="progress-bar "
    role="progressbar"
    style="width: 25%"
    aria-valuenow="25"
    aria-valuemin="0"
    aria-valuemax="100">
  </div>
</div>
<div class="progress">
  <div
    class="progress-bar"
    role="progressbar"
    style="width: 50%"
    aria-valuenow="50"
    aria-valuemin="0"
    aria-valuemax="100">
      50%
  </div>
</div>
<div class="progress">
  <div
    class="progress-bar"
    role="progressbar"
    style="width: 75%"
    aria-valuenow="75"
    aria-valuemin="0"
    aria-valuemax="100">
      75%
  </div>
</div>
<div class="progress">
  <div
    class="progress-bar"
    role="progressbar"
    style="width: 100%"
    aria-valuenow="100"
    aria-valuemin="0"
    aria-valuemax="100">
      100%
  </div>
</div>
"""


s_load_bar = """
<div class="progress">
  <div
    class="progress-bar progress-bar-striped bg-danger"
    role="progressbar"
    style="width: 100%"
    aria-valuenow="100"
    aria-valuemin="0"
    aria-valuemax="100">
  </div>
</div>
"""


result_pred = """
<div
  class="card text-white bg-success mb-3"
  style="max-width: 18rem;">
    <div class="card-body">
      <h5 class="card-title">RESULT</h5>
    </div>
</div>
"""


class0 = """
<div class="alert alert-success" role="alert">
  Result: <b> Malignant </b>
</div>
"""


class1 = """
<div class="alert alert-success" role="alert">
  Result: <b> Benign </b>
</div>
"""

image_uploaded_success = """
<div class="alert alert-success" role="alert">
   Image uploaded successfully!
</div>
"""


more_options = """
<a
  href="#top"
  class="btn btn-info btn-lg active"
  role="button"
  aria-pressed="true"
  style="color:white;">
  More options
</a>
"""


class0_side = """
<div class="alert alert-info" role="alert">
  Result: <b> Malignant tumor </b>
</div>
"""


class1_side = """
<div class="alert alert-info" role="alert">
  Result: <b> Benign tumor </b>
</div>
"""
