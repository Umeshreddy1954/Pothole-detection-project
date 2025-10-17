$(document).ready(function () {
  console.log("Document ready");

  let isLogin = true;

  function showSection(id) {
    $("#home-section, #about-section, #deeplearning-section, #opencv-section").hide();
    $(id).show();
  }

  function setLoggedInUI() {
    $("#auth-section").hide();
    $("#main-nav").show();
    $("#content-section").show();
    showSection("#home-section");

    $(".nav-link").removeClass("active");
    $("#nav-home").addClass("active");
  }

  // Toggle between Login and Register
  $("#toggle-form").click(() => {
    isLogin = !isLogin;
    $("#form-title").text(isLogin ? "Login" : "Register");
    $("#submit-btn").text(isLogin ? "Login" : "Register");
    $("#toggle-form").text(isLogin ? "Register" : "Login");
    $("#confirm-password").toggle(!isLogin);
    $("#email, #password, #confirm-password").val('');
    $("#auth-msg").text('');
  });

  // Show/hide password
  $("#show-password").on("change", function () {
    const type = this.checked ? "text" : "password";
    $("#password").attr("type", type);
    $("#confirm-password").attr("type", type);
  });

  // Handle Login / Registration
  $("#auth-form").submit(function (e) {
    e.preventDefault();
    console.log("Form submitted");

    const email = $("#email").val();
    const password = $("#password").val();
    const confirmPassword = $("#confirm-password").val();

    const url = isLogin ? "/login" : "/api/register";
    const data = isLogin ? { email, password } : { email, password, confirmPassword };

    $("#loader").show();

    $.ajax({
      method: "POST",
      url,
      contentType: "application/json",
      data: JSON.stringify(data),
      success: function (res) {
        $("#loader").hide();
        if (res.status === "success") {
          if (isLogin) {
            setLoggedInUI();
          } else {
            $("#auth-msg").text("Registered! Now login.").css("color", "green");
            $("#email, #password, #confirm-password").val('');
          }
        } else {
          $("#auth-msg").text(res.message).css("color", "red");
        }
      },
      error: function (err) {
        $("#loader").hide();
        $("#auth-msg").text(err.responseJSON?.message || "Error occurred").css("color", "red");
      }
    });
  });

  // Navigation
  $(".nav-link").click(function () {
    const id = $(this).attr("id").replace("nav-", "");
    showSection(`#${id}-section`);
    $(".nav-link").removeClass("active");
    $(this).addClass("active");

    // Clear message on OpenCV section enter
    if (id === "opencv") {
      $("#opencv-msg").text("");
    }
  });

  // Logout
  $("#logout-btn").click(() => {
    $.post("/api/logout", () => {
      location.reload();
    });
  });

  // Image Upload & Prediction
  $("#upload-form").submit(function (e) {
    e.preventDefault();
    const formData = new FormData(this);
    $("#loader").show();

    $.ajax({
      method: "POST",
      url: "/api/upload",
      data: formData,
      processData: false,
      contentType: false,
      success: (res) => {
        $("#loader").hide();
        if (res.status === "success") {
          $("#resnet-result").text(`${res.resnet.label} (${res.resnet.conf}%)`);
          $("#vgg-result").text(`${res.vgg.label} (${res.vgg.conf}%)`);
          $("#yolo-results").empty();
          res.yolo.forEach(label => {
            $("#yolo-results").append(`<li>${label}</li>`);
          });
          $("#result-image").attr("src", `static/predicted/${res.image}`);
          $("#results").fadeIn();
        }
      },
      error: () => {
        $("#loader").hide();
        alert("Upload failed.");
      }
    });
  });

  // Start OpenCV Detection button handler
  $("#start-opencv-btn").click(() => {
    $("#opencv-msg").text("Starting OpenCV detection...");
    $.get("/start-opencv")
      .done(() => {
        $("#opencv-msg").text("OpenCV detection started.").css("color", "green");
      })
      .fail(() => {
        $("#opencv-msg").text("Failed to start OpenCV detection.").css("color", "red");
      });
  });

  // Check session on load
  window.addEventListener("load", () => {
    fetch('/check_session', {
      method: 'GET',
      credentials: 'include'
    })
    .then(res => {
      if (!res.ok) throw new Error("Not logged in");
      return res.json();
    })
    .then(data => {
      console.log("Logged in as", data.username);
      setLoggedInUI();
    })
    .catch(() => {
      console.log("Not logged in");
      $("#auth-section").show();
      $("#home-section").hide();
    });
  });
});
