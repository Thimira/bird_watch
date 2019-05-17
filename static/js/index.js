$(function() {
    function setCorrectness(correctness) {
        var prediction_id = $("#prediction_id").html();
        $.ajax({
            type: "POST",
            url: "/correctness",
            data: JSON.stringify({ 
                prediction_id: prediction_id,
                correctness: correctness
            }),
            contentType: "application/json; charset=utf-8",
            dataType: "json",
            success: function(data) {
                $('#correct-feedback').toast('show')
            },
            failure: function(errMsg) {
                // alert(errMsg);
            }
        });
    }

    $("#pred-correct").click( function(event) {
        event.preventDefault();
        setCorrectness(1)
    });

    $("#pred-incorrect").click( function(event) {
        event.preventDefault();
        setCorrectness(0)
    });

    $("#pred-notsure").click( function(event) {
        event.preventDefault();
        setCorrectness(-1)
    });
});