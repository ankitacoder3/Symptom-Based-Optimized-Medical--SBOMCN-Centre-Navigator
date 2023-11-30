document.addEventListener("DOMContentLoaded", function () {
    const loginBtn = document.getElementById("loginBtn");
    const signupBtn = document.getElementById("signupBtn");
    const loginForm = document.getElementById("loginForm");
    const signupForm = document.getElementById("signupForm");

    loginBtn.addEventListener("click", function () {
        loginBtn.classList.add("active");
        signupBtn.classList.remove("active");
        loginForm.style.display = "block";
        signupForm.style.display = "none";
    });

    signupBtn.addEventListener("click", function () {
        signupBtn.classList.add("active");
        loginBtn.classList.remove("active");
        signupForm.style.display = "block";
        loginForm.style.display = "none";
    });
});
