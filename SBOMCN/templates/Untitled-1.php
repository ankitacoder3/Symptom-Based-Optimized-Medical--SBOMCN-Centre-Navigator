<!--this is for submit. add this to the form div.-->
<form action="process_form.php" method="POST">
    <!-- ... (dropdowns and submit button code) ... -->
</form>




<!--php to collect the options fromm submit and nexxt -->
<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $symptom1 = $_POST["option1"];
    $symptom2 = $_POST["option2"];
    $symptom3 = $_POST["option3"];
    $symptom4 = $_POST["option4"];
    $symptom5 = $_POST["option5"];

    // Process the selected options as needed
    // Example: You can perform database operations, calculations, etc.
    // For now, let's just echo the selected options

    echo "Symptom 1: $symptom1<br>";
    echo "Symptom 2: $symptom2<br>";
    echo "Symptom 3: $symptom3<br>";
    echo "Symptom 4: $symptom4<br>";
    echo "Symptom 5: $symptom5<br>";
}
?>

<!-- Output-->
<!-- ... (existing form code) ... -->
<div id="output"></div>


<!--add this too-->

<script>
    document.addEventListener("DOMContentLoaded", function() {
        const form = document.querySelector("form");
        const outputDiv = document.getElementById("output");

        form.addEventListener("submit", function(event) {
            event.preventDefault(); // Prevent default form submission

            // Collect form data
            const formData = new FormData(form);

            // Send form data to the backend using AJAX
            fetch("process_form.php", {
                method: "POST",
                body: formData
            })
            .then(response => response.text()) // Convert response to text
            .then(data => {
                // Update the output div with the received data
                outputDiv.innerHTML = data;
            })
            .catch(error => {
                console.error("Error:", error);
            });
        });
    });
</script>
<!--this above is to display the ouput in the pridect page. the discese name.-->



<!-- for hospotal page -->


const searchForm = document.getElementById("hospitalSearchForm");
searchForm.addEventListener("submit", function(event) {
    event.preventDefault();
    
    const diseaseName = document.getElementById("diseaseName").value;
    const filter = document.getElementById("filter").value;

    // Perform AJAX request to the backend to get hospital details based on search criteria
    // For demonstration, let's assume you receive a response with an array of hospital objects
    const hospitalDetails = [
        { name: "Hospital A", location: "City X", rating: "4.5" },
        { name: "Hospital B", location: "City Y", rating: "4.2" },
        // ... other hospital objects
    ];

    // Generate HTML to display hospital details
    const hospitalList = document.createElement("ul");
    hospitalDetails.forEach(hospital => {
        const hospitalItem = document.createElement("li");
        hospitalItem.textContent = `${hospital.name} - ${hospital.location} (${hospital.rating} stars)`;
        hospitalList.appendChild(hospitalItem);
    });

    // Replace placeholder content with generated hospital details
    const container = document.querySelector(".container");
    container.innerHTML = ""; // Clear existing content
    container.appendChild(hospitalList);
});
